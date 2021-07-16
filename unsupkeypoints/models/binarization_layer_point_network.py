from .point_network import PointNetwork
import torch


class BinarizationPointNetwork(PointNetwork):
    def __init__(self, parameters, classifier, regressor, criterion):
        self._classifier = classifier
        self._regressor = regressor
        super().__init__(parameters, criterion)

    def forward(self, x):
        x = self._classifier(x)
        probability = torch.sigmoid(x)
        a = torch.bernoulli(probability)
        c = (a - probability).detach()
        x = c + probability
        x = self._regressor(x)
        return x
