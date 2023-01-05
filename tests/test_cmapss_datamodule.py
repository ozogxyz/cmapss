from pathlib import Path

import pytest

from src.datamodules.cmapss_datamodule import CMAPSSDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_cmapss_datamodule(batch_size: int):
    data_dir = "/Users/oozoglu/.rul-datasets"
    print(data_dir)

    dm = CMAPSSDataModule(data_dir=data_dir, fd=1, batch_size=batch_size)
    dm.prepare_data()

    assert Path(data_dir, "CMAPSS").exists()
    assert Path(data_dir, "CMAPSS", "train_FD001.txt").exists()
    assert Path(data_dir, "CMAPSS", "train_FD001.txt").exists()
    assert Path(data_dir, "CMAPSS", "train_FD001.txt").exists()

    dm.setup()
    assert dm.data
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    assert len(dm.data["dev"][0]) == 13818
    assert len(dm.data["val"][0]) == 3913
    assert len(dm.data["test"][0]) == 100

    assert len(dm.data["dev"][1]) == 13818
    assert len(dm.data["val"][1]) == 3913
    assert len(dm.data["test"][1]) == 100

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
