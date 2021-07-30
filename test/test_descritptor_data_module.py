import unittest
import os
from unsupkeypoints.data import DescriptorDataModule
import torch


class TestSevenScenesDataModule(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "data", "test_feature_extractor")
        self._data_module = DescriptorDataModule(dataset_folder, dataset_folder)

    def test_load(self):
        self.assertEqual(len(self._data_module._train_dataset), 255)
        self.assertEqual(len(self._data_module._test_dataset), 255)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["descriptor"].shape, torch.Size([64, 512]))
            self.assertEqual(batch["point3d"].shape, torch.Size([64, 3]))
            self.assertEqual(batch["keypoint"].shape, torch.Size([64, 2]))
            self.assertEqual(batch["position"].shape, torch.Size([64, 4, 4]))
            self.assertEqual(batch["position"].dtype, torch.float32)
            self.assertEqual(batch["keypoint"].dtype, torch.float32)
            break
