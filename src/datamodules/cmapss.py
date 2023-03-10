from rul_datasets.core import RulDataModule
from rul_datasets.reader.cmapss import CmapssReader


class CMAPSSDataModule(RulDataModule):
    """Inherits from :class:`~rul_datasets.core.RulDataModule`
    CMAPSS_URL = https://kr0k0tsch.de/rul-datasets/CMAPSSData.zip
    Select features according to https://doi.org/10.1016/j.ress.2017.11.021"""

    def __init__(self, data_dir: str, fd: int, batch_size: int) -> None:
        # change data download destination from ~/.rul_datasets to project
        # print(rul_datasets.reader.data_root._DATA_ROOT)

        super().__init__(reader=CmapssReader(fd=fd), batch_size=batch_size)  # type: ignore

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @property
    def num_features(self) -> int:
        """Regression problem so we expect a number of cycles of remaining useful life (RUL) as a
        prediction."""

        return 1


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "cmapss.yaml")

    cfg.data_dir = str(root / "data")

    _ = hydra.utils.instantiate(cfg)
