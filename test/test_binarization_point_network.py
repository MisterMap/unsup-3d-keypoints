import torch
import unittest
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.parsing import AttributeDict

from unsupkeypoints.criterions import RegressionLoss
from unsupkeypoints.data import DescriptorDataModule
from unsupkeypoints.models import BinarizationPointNetwork, SimpleModel
from unsupkeypoints.utils.data_module_mock import DataModuleMock
from unsupkeypoints.utils import UniversalFactory


# noinspection PyTypeChecker
class TestBinarizationPointNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "data", "test_feature_extractor")
        self._data_module = DataModuleMock(DescriptorDataModule(dataset_folder, dataset_folder))
        parameters = AttributeDict(
            name="BinarizationPointNetwork",
            optimizer=AttributeDict(),
            classifier=AttributeDict(
                name="SimpleModel",
                input_dimension=512,
                hidden_dimensions=(128,),
                output_dimension=64
            ),
            regressor=AttributeDict(
                name="SimpleModel",
                input_dimension=64,
                hidden_dimensions=(64,),
                output_dimension=3
            ),
            criterion=AttributeDict(
                name="RegressionLoss"
            ),
            metric_logging_frequency=1.
        )
        self._trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        factory = UniversalFactory([RegressionLoss, SimpleModel, BinarizationPointNetwork])
        self._model = factory.make_from_parameters(parameters)

    def test_training(self):
        self._trainer.fit(self._model, self._data_module)

    def test_testing(self):
        self._trainer.test(self._model, self._data_module.test_dataloader())