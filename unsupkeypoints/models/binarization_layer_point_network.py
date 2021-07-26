from .base_lightning_module import BaseLightningModule
import torch


class BinarizationPointNetwork(BaseLightningModule):
    def __init__(self, parameters, classifier, regressor, criterion):
        super().__init__(parameters)
        self._classifier = classifier
        self._regressor = regressor
        self._criterion = criterion

    def forward(self, x):
        x = self._classifier(x)
        probability = torch.sigmoid(x)
        a = torch.bernoulli(probability)
        c = (a - probability).detach()
        x = c + probability
        x = self._regressor(x)
        return x

    def loss(self, batch):
        output = self.forward(batch["descriptor"])

        loss = self._criterion(output, batch["keypoint"], batch["point3d"], batch["position"], batch["mask"])
        return output, loss