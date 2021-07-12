import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .descriptor_dataset import DescriptorDataset


class DescriptorDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, seed=0, batch_size=64, num_workers=4):
        super().__init__()
        torch.manual_seed(seed)
        self._train_dataset = DescriptorDataset(train_path)
        self._test_dataset = DescriptorDataset(test_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        print(f"[DescriptorDataModule] - train dataset size {len(self._train_dataset)}")
        print(f"[DescriptorDataModule] - test dataset size {len(self._test_dataset)}")

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self._train_dataset, self._batch_size, True, pin_memory=False, num_workers=self._num_workers)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self._test_dataset, self._batch_size, False, pin_memory=False, num_workers=self._num_workers)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self._test_dataset, self._batch_size, False, pin_memory=False, num_workers=self._num_workers)

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
