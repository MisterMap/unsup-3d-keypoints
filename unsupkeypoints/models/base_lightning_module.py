import pytorch_lightning as pl
import torch
from ..utils.result_saver import ResultSaver


class BaseLightningModule(pl.LightningModule):
    def __init__(self, parameters):
        super().__init__()
        self.save_hyperparameters(parameters)
        self._result_saver = ResultSaver()

    def loss(self, batch):
        raise NotImplementedError()

    def metric(self):
        metrics = self._result_saver.get_metrics()
        return metrics

    def on_validation_epoch_start(self):
        self._result_saver.clear()

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics()
        self.log_dict(metrics)

    def on_test_epoch_start(self):
        self._result_saver.clear()

    def on_test_epoch_end(self) -> None:
        metrics = self.metrics()
        self.log_dict(metrics)

    def training_step(self, batch, batch_index):
        output, losses = self.loss(batch)
        train_losses = {}
        for key, value in losses.items():
            train_losses[f"train_{key}"] = value
        self.log_dict(train_losses)
        return losses["loss"]

    def validation_step(self, batch, batch_index):
        output, losses = self.loss(batch)
        self._result_saver.save(output, batch)
        val_losses = {}
        for key, value in losses.items():
            val_losses[f"val_{key}"] = value
        self.log_dict(val_losses)
        return losses["loss"]

    def test_step(self, batch, batch_index):
        output, losses = self.loss(batch)
        self._result_saver.save(output, batch)
        val_losses = {}
        for key, value in losses.items():
            val_losses[f"test_{key}"] = value
        self.log_dict(val_losses)
        return losses["loss"]

    def configure_optimizers(self):
        if "betas" in self.hparams.optimizer.keys():
            beta1 = float(self.hparams.optimizer.betas.split(" ")[0])
            beta2 = float(self.hparams.optimizer.betas.split(" ")[1])
            self.hparams.optimizer.betas = (beta1, beta2)
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer)
        if "scheduler" in self.hparams.keys():
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.scheduler)
            return [optimizer], [scheduler]
        return optimizer
