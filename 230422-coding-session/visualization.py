import torch
import torchvision

import matplotlib.pyplot as plt


def visualize(images: torch.Tensor):
    # (batch_size, height, width)
    images_grid = torchvision.utils.make_grid(images, nrow=5)
    # (height, width, batch_size)
    reshaped_images_grid = images_grid.permute(1, 2, 0)
    # Convert tensor to numpy array and plot
    plt.imshow(reshaped_images_grid)
    plt.axis('off')
    plt.show()
