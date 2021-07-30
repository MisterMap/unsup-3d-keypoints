from .base_lightning_module import BaseLightningModule
import torch
import torch.autograd


class BinarizationPointNetwork(BaseLightningModule):
    def __init__(self, parameters, classifier, regressor, criterion, probabilistic=True, sigmoid_after_classifier=True):
        super().__init__(parameters)
        self._classifier = classifier
        self._regressor = regressor
        self._criterion = criterion
        self._probabilistic = probabilistic
        self._sigmoid_after_classifier = sigmoid_after_classifier

    def forward(self, x):
        x = self._classifier(x)
        probability = torch.sigmoid(x)
        if self._probabilistic and self.training:
            a = torch.bernoulli(probability).detach()
            c = (a - probability).detach()
            x = c + probability
        elif self._probabilistic and not self.training:
            x = torch.where(probability > 0.5, 1., 0.)
        elif self._sigmoid_after_classifier:
            x = probability
        x = self._regressor(x)
        return x, probability

    def loss(self, batch):
        output = self.forward(batch["descriptor"])
        loss = self._criterion(output[0], batch["keypoint"], batch["point3d"], batch["position"])
        return output, loss

    @staticmethod
    def calculate_score(probability):
        return torch.mean(torch.abs(probability - 0.5), dim=1)

    def save_result(self, output, batch):
        result = {
            "keypoint": batch["keypoint"],
            "image_index": batch["image_index"],
            "position": batch["position"],
            "predicted_point3d": output[0],
            "point3d": batch["point3d"],
            "score": self.calculate_score(output[1])
        }
        self._result_saver.save(result)
