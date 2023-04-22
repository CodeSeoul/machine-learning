import torch

from torch import nn


class MnistNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._width = 28
        self._height = 28
        self.number_of_classes = 10
        self.network = nn.Sequential(
            nn.Linear(self._width * self._height, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.number_of_classes)
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(input_tensor)
        