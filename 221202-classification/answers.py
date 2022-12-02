"""
If possible, don't look at these while working through the exercises.
Think about each process while working through the jupyter notebook exercises.
Only look at this as a last resort.
"""
# Because I work on a M1 max macbook, need this to prevent PyTorch from crashing
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from tqdm import tqdm

from preprocess import get_mnist_dataloader

# Preprocess
# ---------------------
batch_size: int = 32
training_dataloader = get_mnist_dataloader(batch_size, is_training_dataset=True)
test_dataloader = get_mnist_dataloader(batch_size, is_training_dataset=False)

# Define models
# -------------------------
# see MnistClassifier for more info on how to define PyTorch models.
from models import MnistClassifier

width, height = 28, 28
flattened_dimensions: int = width * height
num_classes: int = 10
model = MnistClassifier([(flattened_dimensions, 512), (512, 256), (256, num_classes)])

# Test the model to see
test_tensor = torch.randn((batch_size, flattened_dimensions))
output = model(test_tensor)

# Should be (batch_size, num_classes) =  torch.Size([32, 10])
print(f'Model output: {output.shape}')

# Add optimizer and loss function
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
cross_entropy_loss = torch.nn.CrossEntropyLoss()


@torch.no_grad()
def get_model_accuracy(model_to_evaluate: torch.nn.Module,
                       evaluation_dataloader: torch.utils.data.DataLoader) -> float:
    """
    Given a model and dataloader, evaluate the accuracy of the model on the target dataset.
    Args:
        model_to_evaluate: The model to evaluate
        evaluation_dataloader: The evaluation dataset

    Returns:
        The accuracy of the model
    """
    model_to_evaluate.eval()
    total_correct = 0
    total_number_of_samples = 0
    for x_eval, y_eval in evaluation_dataloader:
        # see training loop for more information
        x_eval = x_eval.flatten(start_dim=1)
        y_pred = model_to_evaluate(x_eval)
        # Now, during inference, we want to pick the argmax
        y_pred = torch.argmax(y_pred, dim=1)
        # The correct predictions are ones that are correct with the labels
        # (y_pred == y_eval) will yield a tensor with true if values are equal, false otherwise.
        # If we call sum on boolean tensors, it casts the boolean tensors to int.
        # By adding these up we get the number of correct labels
        # Item retrieves the value stored in the tensor.
        # Since sum reduces the values into a single numerical value, we can immediately
        # retrieve that value right away
        correct = (y_pred == y_eval).sum().item()
        total_correct += correct
        # Add batch_size
        total_number_of_samples += x_eval.shape[0]
    model_to_evaluate.train()
    return total_correct / total_number_of_samples


# training loop
# -------------------------------
epochs = 10

# Before that, lets take a look the shape of each of the elements
x_train, y_train = next(iter(training_dataloader))

# (batch_size, channel, height, width) -> (32, 1, 28, 28)
print(f'x_train shape: {x_train.shape}')
# (batch_size) -> (32)
print(f'y_train shape: {y_train.shape}')
# each entry contains the corresponding label ranging from 0 - 9
print(y_train)

# Set model to training mode. In large applications, please do this, because
# in most cases, we do not know what kind of processing may have been done with
# the model prior to the training loop. This will save you from a lot of headaches.
model.train()

for epoch in range(epochs):
    # x_train contains batch of training images
    # y_train contains batch of labels pertaining to each of the training images
    for i, (x_train, y_train) in tqdm(enumerate(training_dataloader)):
        # reset all accumulated gradients
        optimizer.zero_grad()

        # Reshape from (batch_size, channel, height, width) to (batch_size, channel * height * width)
        # = (32, 1, 28, 28) -> (32, 1 * 28 * 28) = (32, 784)
        # flatten across second dimension (channel) as follows:
        x_train = x_train.flatten(start_dim=1)
        # we can also achieve this manually via .reshape() or .view()
        # as follows
        # x_train = x_train.view(batch_size, -1)
        # or
        # x_train = x_train.reshape(batch_size, -1)
        # -1 means fit whatever is remaining across this dimension
        # in this case, it is the channel dimension or dimension index=1

        # Now, feed this into the model
        # shaped: (batch_size, number of classes) -> (32, 10)
        y_pred: torch.Tensor = model(x_train)

        # Calculate loss
        # First arg - y_pred,
        # then ground-truth
        loss = cross_entropy_loss(y_pred, y_train)

        # Back-propagation
        loss.backward()

        # Update model weights
        optimizer.step()

        if i % 300 == 0:
            print(f'epoch [{epoch + 1} / {epochs}], step: [{i}/{len(training_dataloader)}]: Loss: {loss}')

    # Training accuracy
    training_accuracy = get_model_accuracy(model, training_dataloader)
    training_accuracy = training_accuracy * 100
    print(f'Epoch {epoch + 1} model training accuracy: {training_accuracy:.2f}%')

    # Evaluation
    # -----------------------
    # Note: For each of these operations, in the real-world, try to keep each unit of operation in a single function
    # to enable testing
    test_accuracy = get_model_accuracy(model, test_dataloader)
    test_accuracy = test_accuracy * 100
    print(f'Epoch {epoch + 1} model test accuracy: {test_accuracy:.2f}%')


