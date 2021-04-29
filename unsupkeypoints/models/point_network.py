import torch.nn as nn

from .base_lightning_module import BaseLightningModule


class PointNetwork(BaseLightningModule):
    def __init__(self, parameters, criterion):
        super().__init__(parameters)
        self._criterion = criterion
        self.backend = self.make_backend(parameters)

    @staticmethod
    def make_backend(parameters):
        modules = []
        input_dimension = parameters.input_dimension
        output_dimension = 3
        for dimension in parameters.hidden_dimensions:
            modules.append(nn.Linear(input_dimension, dimension))
            input_dimension = dimension
            modules.append(nn.ReLU())
        modules.append(nn.Linear(input_dimension, output_dimension))
        return nn.Sequential(*modules)

    def forward(self, x):
        return self.backend(x)

    def loss(self, batch):
        output = self.forward(batch["descriptor"])

        loss = self._criterion(output, batch["keypoint"], batch["point3d"], batch["position"], batch["mask"])
        return output, loss
