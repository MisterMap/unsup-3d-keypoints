from .base_lightning_module import BaseLightningModule
import torch
import torch.autograd


class BinarizationPointNetwork(BaseLightningModule):
    def __init__(self, parameters, classifier, regressor, criterion, probabilistic=True):
        super().__init__(parameters)
        self._classifier = classifier
        self._regressor = regressor
        self._criterion = criterion
        self._probabilistic = probabilistic

    def forward(self, x):
        x = self._classifier(x)
        probability = torch.sigmoid(x)
        if self._probabilistic and self.training:
            a = torch.bernoulli(probability).detach()
            c = (a - probability).detach()
            x = c + probability
        elif self._probabilistic:
            x = torch.where(probability > 0.5, 1., 0.)
        else:
            x = probability
        x = self._regressor(x)
        return x

    def loss(self, batch):
        output = self.forward(batch["descriptor"])

        loss = self._criterion(output, batch["keypoint"], batch["point3d"], batch["position"], batch["mask"])
        return output, loss