from rul_datasets.reader.cmapss import CmapssReader
from rul_datasets.core import RulDataModule


class CMAPSSDataModule(RulDataModule):
    """Inherits from :class:`~rul_datasets.core.RulDataModule`
    CMAPSS_URL = https://kr0k0tsch.de/rul-datasets/CMAPSSData.zip
    Select features according to https://doi.org/10.1016/j.ress.2017.11.021"""

    def __init__(self, data_dir: str, fd: int, batch_size: int) -> None:
        # change data download destionation from ~/.rul_datasets to project
        # print(rul_datasets.reader.data_root._DATA_ROOT)

        super().__init__(reader=CmapssReader(fd=1), batch_size=batch_size)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @property
    def num_features(self) -> int:
        """Regression problem so we expect a number of cycles of remaining useful life (RUL) as a prediction."""

        return 1


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "cmapss.yaml")

    cfg.data_dir = str(root / "data")

    _ = hydra.utils.instantiate(cfg)
