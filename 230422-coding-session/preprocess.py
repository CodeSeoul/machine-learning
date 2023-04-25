from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms


def get_mnist_dataloader(batch_size: int,
                         is_training_dataset: bool) -> DataLoader:
    """
    Retrieve Mnist dataloader
    Args:
        batch_size: The batch size during training and evaluation
        is_training_dataset: Set to true to retrieve training set. Otherwise, return False
    Returns:
        A dataloader object
    """
    mnist_dataset = MNIST(root='.', download=True, train=is_training_dataset, transform=transforms.Compose([
        # Convert PIL to tensor and normalize values between 0 and 1 by dividing all pixels by 255
        transforms.ToTensor(),
        # Place tensor values from 0 - 1 to -1 - 1
        transforms.Normalize([0.5], [0.5])

    ]))
    return DataLoader(mnist_dataset,
                      # Size of mini-batch
                      batch_size=batch_size,
                      # Shuffle data each time during training to randomize samples
                      shuffle=True,
                      # We want to drop the last few to ensure that our batch_size remains constant.
                      drop_last=True,
                      # Pin memory speeds up the host to device (usually gpu in the real-world)
                      # during training
                      # Ideally, having all data on the GPU makes training faster, but most GPUs do not have enough
                      # memory to host an entire dataset, so during training, we need to move the tensors from
                      # cpu -> gpu to speed up operations
                      pin_memory=True)


def get_cifar_dataloader(batch_size: int,
                         is_training_dataset: bool) -> DataLoader:
    """
    Retrieve Mnist dataloader
    Args:
        batch_size: The batch size during training and evaluation
        is_training_dataset: Set to true to retrieve training set. Otherwise, return False
    Returns:
        A dataloader object
    """
    cifar_dataset = CIFAR10(root='.', 
                            download=True, 
                            train=is_training_dataset, 
                            transform=transforms.Compose([
        # Convert PIL to tensor and normalize values between 0 and 1 by dividing all pixels by 255
        transforms.ToTensor(),
        # Place tensor values from 0 - 1 to -1 - 1
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ]))
    return DataLoader(cifar_dataset,
                      # Size of mini-batch
                      batch_size=batch_size,
                      # Shuffle data each time during training to randomize samples
                      shuffle=True,
                      # We want to drop the last few to ensure that our batch_size remains constant.
                      drop_last=True,
                      # Pin memory speeds up the host to device (usually gpu in the real-world)
                      # during training
                      # Ideally, having all data on the GPU makes training faster, but most GPUs do not have enough
                      # memory to host an entire dataset, so during training, we need to move the tensors from
                      # cpu -> gpu to speed up operations
                      pin_memory=True)
