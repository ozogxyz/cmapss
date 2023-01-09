from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
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
        self.save_hyperparameters(logger=False, ignore=["net"])

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
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_rmse_best.reset()

    def _step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        # reshape targets from [batch_size] to [batch_size, pred.size(1)] for correct loss
        # y = y.view(y.size(0), preds.size(1))
        # print(y.shape, preds.shape)
        loss = self.criterion(preds, y)
        self.train_loss(loss.item())
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_rmse(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_rmse(preds, targets)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=False
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # get current val rmse
        rmse = self.val_rmse.compute()
        # update the best val rmse so far
        self.val_rmse_best(rmse)
        # log `val_rmse_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/rmse_best", self.val_rmse_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_rmse(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in the optimization.
        Normally we'd need one. But in the case of GANs or similar we might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
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
