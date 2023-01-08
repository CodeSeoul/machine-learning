import random

import numpy as np

from dataset import get_data
import matplotlib.pyplot as plt


class SimpleLinearRegression:
    def __init__(self, learning_rate: float = 0.0001):
        # theta_1
        self.weights = random.random()
        # theta_0
        self.bias = random.random()
        self.learning_rate = learning_rate

    def l2_loss(self, ground_truth: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Given the ground truth (labels) and the predictions outputted by our model,
        calculate the L2 loss
        Args:
            ground_truth: The labels provided in the dataset
            y_pred: The corresponding prediction made by the model

        Returns:
            The l2 or mean squared error loss
        """
        raise NotImplemented

    def _gradient_descent(self, x_train, y_train, y_pred) -> None:
        """
        Given a loss vector, update the parameters of the model W.R.T the loss.
        """
        raise NotImplemented

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 5000) -> None:
        """

        Args:
            x_train: The training dataset (input features)
            y_train: The ground truth labels
            epochs: The number of epochs to train for

        Returns:
            Nothing. The weights of this model is fitted to the given dataset.
        """
        raise NotImplemented

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given the input dataset, map the features to prediction.
        A simple linear regression model models the following equation

        Y = ax + b
        Args:
            X: The input data (batch, 1) dim

        Returns:
            The output of F(x)
        """
        raise NotImplemented

    def visualize(self, X: np.ndarray, Y: np.ndarray, y_pred) -> None:
        # Plot the dataset (dots)
        plt.scatter(X, Y)
        # Draw line of best fit
        plt.plot(X, y_pred)
        plt.show()


def do_linear_regression():
    # Step 1. Retrieve the data.
    x_train, y_train = get_data('Salary_Data.csv')
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}')

    # Step 2. Define model
    # Before proceeding, make sure that SimpleLinearRegression is Defined
    model = SimpleLinearRegression(learning_rate=0.001)

    # 3. Train the model
    model.fit(x_train, y_train)

    # 4. Try making predictions
    y_pred = model.predict(x_train)

    # 5. Visualize the data
    model.visualize(x_train, y_train, y_pred)


if __name__ == '__main__':
    do_linear_regression()
