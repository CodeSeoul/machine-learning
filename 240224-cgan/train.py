import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define all of our imports here
import typing as t

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# For visualization
from tqdm import tqdm

from preprocess import get_gpu_device_safe, get_mnist_dataloader
from models import get_discriminator_and_generator
from util import generate_fake_images, print_stats, visualize_loss, visualize_tensors


def main():
    # Training
    batch_size: int = 32
    epochs: int = 10
    # index of GPU if available
    # In actual applications, don't hard code this value, but use something like argparse or hydra
    gpu_index: int = 0
    # 0.01
    learning_rate: float = 1e-4
    latent_dim: int = 100
    logging_interval: int = 300

    # Dataloader
    training_dataloader = get_mnist_dataloader(batch_size, is_training_dataset=True)
    # check device to see whether it is available
    device = get_gpu_device_safe(torch.device(f'cuda:{gpu_index}'))

    # Define models
    discriminator, generator = get_discriminator_and_generator(latent_dim, device)

    # train the model
    train(generator, discriminator, training_dataloader, batch_size, latent_dim, device, epochs, learning_rate,
          logging_interval)


def train_generator_and_get_loss(generator: nn.Module,
                                 discriminator: nn.Module,
                                 batch_size: int,
                                 latent_dim: int,
                                 device: t.Union[torch.device, int],
                                 gt_real: torch.Tensor,
                                 labels: torch.Tensor,
                                 optimizer: torch.optim.Optimizer) -> torch.Tensor:
    # Train generator
    # ----------------------------

    # Clear all previously accumulated gradients
    optimizer.zero_grad()

    fake_images = generate_fake_images(generator, labels, batch_size, latent_dim, device)
    discriminator_fake = discriminator(fake_images, labels)

    # Loss is minimized by fooling the discriminator into thinking the image is real
    generator_loss = F.binary_cross_entropy(discriminator_fake, gt_real)

    # Backprop
    generator_loss.backward()
    optimizer.step()

    return generator_loss


def train_discriminator_and_get_loss(generator: nn.Module,
                                     discriminator: nn.Module,
                                     real_images: torch.Tensor,
                                     batch_size: int,
                                     latent_dim: int,
                                     device: t.Union[torch.device, int],
                                     gt_real: torch.Tensor,
                                     gt_fake: torch.Tensor,
                                     labels: torch.Tensor,
                                     optimizer: torch.optim.Optimizer
                                     ) -> torch.Tensor:
    # Set all accumulated gradients to zero
    optimizer.zero_grad()

    # we are not accumulating gradients here since we are training discriminator
    with torch.no_grad():
        fake_images = generate_fake_images(generator, labels, batch_size, latent_dim, device).detach()

    # combine the real and fake images into a single batch
    discriminator_fake = discriminator(fake_images, labels)
    discriminator_real = discriminator(real_images, labels)

    # Calculate loss: divide by 2 since we are doubling the batch size by including fake images
    loss_real = F.binary_cross_entropy(discriminator_real, gt_real)
    loss_fake = F.binary_cross_entropy(discriminator_fake, gt_fake)
    discriminator_loss = (loss_real + loss_fake) / 2

    # Backprop
    discriminator_loss.backward()
    # Update the parameters
    optimizer.step()

    return discriminator_loss


def train(generator: nn.Module,
          discriminator: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          batch_size: int,
          latent_dim: int,
          device: t.Union[torch.device, int],
          epochs: int,
          learning_rate: float,
          logging_interval: int,
          ):
    # Ground-truth labels
    # Dimension: (batch_size, 1)
    gt_real = torch.tensor([0.9] * batch_size, device=device, requires_grad=False).reshape(batch_size, 1)
    gt_fake = torch.tensor([0.1] * batch_size, device=device, requires_grad=False).reshape(batch_size, 1)

    # Optimizer
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                               lr=learning_rate,
                                               betas=(0.5, 0.999),
                                               weight_decay=1e-3)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Image dimensions - MNIST is 28 x 28 gray-scale images
    width, height = 28, 28
    flattened_image_dim = width * height

    discriminator.train()
    generator.train()

    discriminator_loss_per_epoch = []
    generator_loss_per_epoch = []

    for epoch in range(epochs):
        # Loss for plotting
        discriminator_running_loss = 0.0
        generator_running_loss = 0.0
        # The second (y_train or '_') are the class labels which are not used
        # when training the standard GAN
        for current_step, (real_images, labels) in tqdm(enumerate(train_dataloader)):
            # Place on appropriate device
            real_images = real_images.reshape(-1, flattened_image_dim).to(device)

            # Train generator
            generator_loss = train_generator_and_get_loss(
                generator, 
                discriminator, 
                batch_size, 
                latent_dim, 
                device, 
                gt_real, 
                labels,
                generator_optimizer
            )

            # Note: In production code, it is not a good idea to create functions that
            # take in many arguments. Break it down into smaller chunks
            discriminator_loss = train_discriminator_and_get_loss(
                # Networks
                generator, discriminator,
                # Real image for training discriminator
                real_images,
                # Needed to generate batch
                batch_size, latent_dim, device,
                # ground truth
                gt_real, 
                gt_fake,
                labels,
                discriminator_optimizer
            )

            # Calculate running loss
            discriminator_running_loss += discriminator_loss.item()
            generator_running_loss += generator_loss.item()

            # print stats
            if current_step % logging_interval == 0:
                # need to detach from graph before using with third-party visualization libraries
                discriminator_loss = discriminator_loss.cpu().detach().numpy()
                generator_loss = generator_loss.cpu().detach().numpy()
                print_stats(epoch, epochs, current_step, discriminator_loss, generator_loss)

        # After each epoch, visualize image
        with torch.no_grad():
            # Accumulate running loss
            generator_loss_per_epoch.append(generator_running_loss / len(train_dataloader))
            discriminator_loss_per_epoch.append(discriminator_running_loss / len(train_dataloader))

            # change to evaluation mode
            generator.eval()

            # Generate fake images to visualize
            row_count, column_count = 5, 5
            image_count = row_count * column_count
            fake_labels = torch.randint(0, 9, (image_count,), device=device, dtype=torch.long)
            fake_images = generate_fake_images(generator, fake_labels, image_count, latent_dim, device).cpu().detach()

            # Reshape from 784 to 28 x 28 since linear layers output vectors
            fake_images = fake_images.reshape(image_count, width, height)

            # Visualize generated images
            visualize_tensors(fake_images, row_count, column_count, epoch)

            # Revert back to training mode
            generator.train()

    # Lastly, visualize loss function
    visualize_loss(discriminator_loss_per_epoch, generator_loss_per_epoch)


if __name__ == '__main__':
    main()
