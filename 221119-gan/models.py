import typing as t

import torch
import torch.nn as nn

import os
# For my mac which has issues running opencv, PyTorch, without this option.
# Feel free to remove this if you don't have any issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Discriminator(nn.Module):
    def __init__(self, layer_dimensions: t.List[t.Tuple[int, int]]):
        super().__init__()
        self._validate(layer_dimensions)

        self.network = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(0.2, inplace=True)
            )
            for input_dim, output_dim in layer_dimensions
        ])
        self.last_layer_dim = layer_dimensions[-1][-1]
        self.output_layer = nn.Sequential(nn.Linear(self.last_layer_dim, 1), nn.Sigmoid())

    def _validate(self, layer_dimensions: t.List[t.Tuple[int, int]]):
        if not isinstance(layer_dimensions, t.List) and len(layer_dimensions) == 0:
            raise TypeError('layer_dimensions must be a non-empty list')

        for dims in layer_dimensions:
            if not isinstance(dims, t.Tuple) and len(dims) != 2:
                raise TypeError('layer_dimensions must be a list of two-tuples')
            for dim in dims:
                if not isinstance(dim, int):
                    raise TypeError(f'Dimensions must be integer values. Specified: {type(dim)}')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        intermediate_feature = self.network(X)
        return self.output_layer(intermediate_feature)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        # Same as above, but you can write it like this
        # if the code above is confusing.
        self.layer_1 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1024, 784),
            # tanh value ranges from -1 to 1, same as a normalized image
            nn.Tanh()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        first_layer = self.layer_1(X)
        second_layer = self.layer_2(first_layer)
        third_layer = self.layer_3(second_layer)
        return self.output_layer(third_layer)


def get_discriminator_and_generator(latent_dim: int, device: t.Union[torch.device, int]) -> t.Tuple[
    nn.Module, nn.Module]:
    """
    Given the latent dim vector size and device, return the discriminator and generator model
    Args:
        latent_dim: The latent vector dimension
        device: The target device to place models on
    """
    # Width of the mnist images. Again, don't hard code this in real-world
    width, height = 28, 28
    discriminator = Discriminator([
        # First linear layer: (N x 784) -> (N x 256)
        (width * height, 512),
        (512, 256),
        # Last layer is a sigmoid layer
    ]).to(device)
    generator = Generator(latent_dim).to(device)
    return discriminator, generator
