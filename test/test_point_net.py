import os
import unittest

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.parsing import AttributeDict

from unsupkeypoints.criterions import RegressionLoss, RGBandModelReprojectionLoss
from unsupkeypoints.data import DescriptorDataModule
from unsupkeypoints.models import PointNetwork
from unsupkeypoints.utils.data_module_mock import DataModuleMock


# noinspection PyTypeChecker
class TestPointNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "data", "test_feature_extractor")
        self._data_module = DataModuleMock(DescriptorDataModule(dataset_folder, dataset_folder))
        self._params = AttributeDict(
            name="point_net",
            optimizer=AttributeDict(),
            hidden_dimensions=(100,),
            input_dimension=512,
        )
        self._trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        self._criterion = RegressionLoss()
        self._model = PointNetwork(self._params, self._criterion)

    def test_training(self):
        self._trainer.fit(self._model, self._data_module)

    def test_testing(self):
        self._trainer.test(self._model, self._data_module.test_dataloader())


class TestPointNetWithRgbAndModelReprojectionLoss(TestPointNet):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "data", "test_feature_extractor")
        self._data_module = DataModuleMock(DescriptorDataModule(dataset_folder, dataset_folder))
        self._params = AttributeDict(
            name="point_net",
            optimizer=AttributeDict(),
            hidden_dimensions=(100,),
            input_dimension=512,
        )
        self._trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        self._criterion = RGBandModelReprojectionLoss()
        self._model = PointNetwork(self._params, self._criterion)
