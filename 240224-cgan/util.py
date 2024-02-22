import functools
import os.path
import typing as t

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm


def print_stats(current_epoch: int,
                total_epochs: int,
                current_step: int,
                discriminator_loss: float,
                generator_loss: float):
    tqdm.write(f'[epoch {current_epoch} / {total_epochs}]: '
               f'current_step: {current_step}, '
               f'disc_loss: {discriminator_loss:.5f}, '
               f'generator_loss: {generator_loss:.5f}')


def generate_fake_images(generator: nn.Module,
                         gt_real: torch.Tensor,
                         batch_size: int,
                         latent_dim: int,
                         device: t.Union[torch.device, int]) -> torch.Tensor:
    """
    Given a generator model and the batch size, generate a batch of
    "latent_dim" dimensional vector
    Args:
        generator: The generator model to use
        gt_real: The ground truth labels
        batch_size: The random noise batch size
        device: The device to place random noise tensor
        latent_dim: The dimension of the latent vector
    """
    z = torch.randn((batch_size, latent_dim), device=device)
    return generator(z, gt_real)


def make_if_not_exist(path: str) -> t.Callable:
    """
    Simple decorator for creating paths if they dont exist
    Args:
        path: The target path to examine. Will create a directory if it does not exist
    """
    def inner(wrapped_function: t.Callable) -> t.Callable:

        @functools.wraps(wrapped_function)
        def decorator(*args, **kwargs) -> t.Any:
            if not os.path.exists(path):
                os.mkdir(path)
            return wrapped_function(*args, **kwargs)
        return decorator
    return inner


@make_if_not_exist('images')
def visualize_loss(discriminator_loss: t.List, 
                   generator_loss: t.List, 
                   is_jupyter: bool = False):
    plt.clf()
    plt.plot(discriminator_loss, label='discriminator_loss')
    plt.plot(generator_loss, label='generator_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['discriminator_loss', 'generator_loss'], loc='upper right')
    if is_jupyter:
        plt.show()
    else:
        plt.savefig(f'images/loss.png')


@make_if_not_exist('images')
def visualize_tensors(images: torch.Tensor,
                      row_count: int,
                      column_count: int,
                      epoch_no: int,
                      is_jupyter: bool = False):
    figure = plt.figure(figsize=(10, 8))
    one_based_offset = 1
    for i in range(one_based_offset, row_count * column_count + one_based_offset):
        figure.add_subplot(row_count, column_count, i)
        img = images[i - one_based_offset]
        plt.axis("off")
        plt.imshow(img.numpy().squeeze(), cmap="gray")

    if is_jupyter:
        plt.show()
    else:
        plt.savefig(f'images/epoch_{epoch_no}.png')
