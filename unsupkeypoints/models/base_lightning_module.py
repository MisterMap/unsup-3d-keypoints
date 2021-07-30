import pytorch_lightning as pl
import torch
from ..utils.result_saver import ResultSaver


def add_prefix_to_keys(dictionary, prefix):
    result = {}
    for key, value in dictionary.items():
        result[f"{prefix}_{key}"] = value
    return result


class BaseLightningModule(pl.LightningModule):
    def __init__(self, parameters):
        super().__init__()
        self.save_hyperparameters(parameters)
        self._result_saver = ResultSaver()
        self._metric_logging_frequency = parameters.metric_logging_frequency

    def loss(self, batch):
        raise NotImplementedError()

    def metrics(self):
        metrics = self._result_saver.get_metrics()
        return metrics

    def on_train_epoch_start(self) -> None:
        if self.is_metric_logging_epoch():
            self.clear_result_saver()

    def on_train_epoch_end(self, unused=None) -> None:
        if self.is_metric_logging_epoch():
            metrics = self.metrics()
            metrics = add_prefix_to_keys(metrics, "train")
            self.log_dict(metrics)

    def on_validation_epoch_start(self):
        self.clear_result_saver()

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics()
        self.log_dict(metrics)

    def on_test_epoch_start(self):
        self.clear_result_saver()

    def on_test_epoch_end(self) -> None:
        metrics = self.metrics()
        metrics = add_prefix_to_keys(metrics, "test")
        self.log_dict(metrics)

    def training_step(self, batch, batch_index):
        return self.learning_step(batch, batch_index, "train")

    def validation_step(self, batch, batch_index):
        return self.learning_step(batch, batch_index, "val")

    def test_step(self, batch, batch_index):
        return self.learning_step(batch, batch_index, "test")

    def learning_step(self, batch, batch_index, prefix="train"):
        output, losses = self.loss(batch)
        logged_losses = {}
        for key, value in losses.items():
            logged_losses[f"{prefix}_{key}"] = value
        self.log_dict(logged_losses)
        self.save_result(output, batch)
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

    def save_result(self, output, batch):
        result = {
            "keypoint": batch["keypoint"],
            "image_index": batch["image_index"],
            "position": batch["position"],
            "predicted_point3d": output,
            "point3d": batch["point3d"]
        }
        self._result_saver.save(result)

    def clear_result_saver(self):
        self._result_saver.clear()

    def is_metric_logging_epoch(self):
        return self.current_epoch % self._metric_logging_frequency == self._metric_logging_frequency - 1
