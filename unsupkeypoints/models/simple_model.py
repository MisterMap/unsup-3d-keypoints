import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU
}


class SimpleModel(nn.Sequential):
    def __init__(self, input_dimension, output_dimension, hidden_dimensions=tuple(), activation="relu"):
        modules = []
        for dimension in hidden_dimensions:
            modules.append(nn.Linear(input_dimension, dimension))
            input_dimension = dimension
            modules.append(ACTIVATIONS[activation]())
        modules.append(nn.Linear(input_dimension, output_dimension))
        super().__init__(*modules)
