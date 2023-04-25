import torch

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34

from preprocess import get_gpu_device_safe, get_cifar_dataloader
from visualization import visualize

def train(model_to_train: nn.Module,
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          device: torch.device, 
          epochs_to_train: int, 
          print_interval: int) -> None:
    """Train the model on the given dataset

    Args:
        model_to_train (nn.Module): The model that we wish to train
        train_dataloader (DataLoader): The dataloader that we wish to use during training
        test_dataloader (DataLoader): The test dataloader we will be using to evaluate
        device (torch.device): The loss function
        batch_size (int): The number of data points we will train on each step.
        epochs_to_train (int): The number of epochs to train for
        print_interval (int): The number of intervals to train for
    """
    # Define optimizer
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for current_epoch in range(1, epochs_to_train):
        model_to_train.train()
        print(f'Beginning epoch: {current_epoch}')
        for step, (x_train, y_train) in enumerate(tqdm(train_dataloader)):
            
            # Important: need to reset gradients. Otherwise, we will have exploding gradients
            optimizer.zero_grad()

            # Assign tensors to target device. By default, dataloader tensors are stored on CPU,
            # since GPU has limited memory
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            # Prediciton
            y_pred = model_to_train(x_train)

            # Loss
            loss = criterion(y_pred, y_train)
            
            # backprop to accumulate gradients
            loss.backward()
            optimizer.step()

            if step % print_interval == 0:
                print(f'Loss: {loss}')

        accuracy = get_accuracy(model, test_dataloader, device)
        print(f'Accuracy: {accuracy:.2f}%')


@torch.no_grad()
def get_accuracy(model_to_evaluate: nn.Module, 
                 test_dataloader: DataLoader, 
                 device: torch.device) -> float:
    """get the accuracy of the model on the given test dataloader

    Args:
        model_to_evaluate (nn.Module): The model to evaluate
        test_dataloader (DataLoader): The dataloader to use for evaluation
        device (torch.device): The device to run the evaluation on
    """
    model_to_evaluate.eval()
    total_correct = 0
    total = 0
    for x_test, y_test in test_dataloader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        # Prediciton
        y_pred = model_to_evaluate(x_test)
        y_pred = torch.argmax(y_pred, dim=1)

        # Calculate accuracy
        total_correct += (y_pred == y_test).sum().item()
        total += x_test.shape[0]
    
    return (total_correct / total) * 100



train(model, train_dataloader, test_dataloader, device, epochs, print_interval)
