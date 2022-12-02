import typing as t

import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 leak_slope: float = 0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(inplace=True, negative_slope=leak_slope)
        )

    def forward(self, input_data: torch.Tensor):
        return self.layer(input_data)


class MnistClassifier(nn.Module):
    def __init__(self, layer_dims: t.List[t.Tuple[int, int]]):
        super().__init__()
        self._validate_arguments(layer_dims)
        intermediate_layers = [
            LinearLayer(input_dim, output_dim)
            for input_dim, output_dim in layer_dims[:-1]
        ]
        intermediate_layers.append(nn.Linear(*layer_dims[-1]))
        self.layers = nn.Sequential(*intermediate_layers)

    def _validate_arguments(self, layer_dims: t.List[t.Tuple[int, int]]):
        if not isinstance(layer_dims, t.List):
            raise TypeError(f'layer_dims must be a list. Passed in: {layer_dims}')

        if not layer_dims:
            raise ValueError('Cannot pass in an empty list')

        for layer_tuple in layer_dims:
            if not isinstance(layer_tuple, t.Tuple):
                raise TypeError('layer inputs should be a tuple. '
                                f'Passed in: {layer_tuple}')

            if not len(layer_tuple) == 2:
                raise ValueError('layer_dims must be a list of two-tuples')

            for item in layer_tuple:
                if not isinstance(item, int):
                    raise TypeError('Each two-tuple in layer_dims must be an int. '
                                    f'Passed in value: {item}')

    def forward(self, X: torch.Tensor):
        return self.layers(X)


if __name__ == '__main__':
    pass