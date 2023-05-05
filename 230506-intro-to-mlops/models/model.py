import numpy as np
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST


class LitMNIST(LightningModule):
    def __init__(
        self,
        batch_size: int,
        lr: float,
        data_dir: str = 'datasets',
    ):

        super().__init__()
        self.batch_size = batch_size

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.lr = lr

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, 256),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes),
        )

        self.val_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)

    def preprocess(self, input_data: np.ndarray) -> torch.Tensor:
        # (28 x 28) -> (1, 1, 28, 28)
        return self.transform(input_data).view(-1, *self.dims)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        dataset_train = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(dataset_train, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, shuffle=True, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=2)



def get_mnist_pytorch_lightning_trainer(max_epochs: int) -> Trainer:
    """
    Get a simple PyTorch Lightning Trainer
    Args:
        max_epochs: The maximum number of epochs to train for
    """
    return Trainer(
        accelerator='auto',
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )


def train_mnist_model(
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    epochs: int = 10,
) -> LitMNIST:
    """
    Train the MNIST model with the following settings

    Args:
        batch_size: The mini-batch size during training and testing
        learning_rate: The learning rate of the optimizer
        epochs: The name of the experiment

    Returns: 
        The trained model
    """
    model = LitMNIST(
        batch_size=batch_size,
        lr=learning_rate,
    )
    model.setup()

    # Train model
    trainer = get_mnist_pytorch_lightning_trainer(max_epochs=epochs)
    trainer.fit(model)

    return model


if __name__ == "__main__":
    train_mnist_model()
