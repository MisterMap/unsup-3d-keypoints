from .data_loader_mock import DataLoaderMock
import pytorch_lightning as pl


# noinspection PyAbstractClass
class DataModuleMock(pl.LightningDataModule):
    def __init__(self, data_module):
        super().__init__()
        self._data_module = data_module

    def train_dataloader(self):
        return DataLoaderMock(self._data_module.train_dataloader())

    def val_dataloader(self):
        return DataLoaderMock(self._data_module.val_dataloader())

    def test_dataloader(self):
        return DataLoaderMock(self._data_module.test_dataloader())
