from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression.mse import MeanSquaredError


class PLModule(LightningModule):
    """Example of LightningModule for CMAPSS-RUL estimation.

        A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True, ignore=["net"])  # TODO metric name none

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging RMSE across batches
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        # for averaging loss across bathces
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best validation metric (lowest RMSE) so far
        self.val_rmse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self) -> None:
        # reset metrics after sanity check
        self.val_rmse_best.reset()

    def _step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.train_loss(loss.item())
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_rmse(preds, targets)
        self.log_dict(
            {"train/loss": self.train_loss, "train/rmse": self.train_rmse},
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_rmse(preds, targets)
        self.log_dict(
            {"val/loss": self.val_loss, "val/rmse": self.val_rmse},
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        rmse = self.val_rmse.compute()
        self.val_rmse_best(rmse)

        # use `.compute()` otherwise it would be reset by lightning after each epoch
        self.log("val/rmse_best", self.val_rmse_best.compute(), prog_bar=False)


    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_rmse(preds, targets)
        self.log_dict(
            {"test/loss": self.test_loss, "test/rmse": self.test_rmse},
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer: torch.optim.Optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler: Any = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss",
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "cmapss.yaml")
    _ = hydra.utils.instantiate(cfg)
