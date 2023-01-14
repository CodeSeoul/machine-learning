import random

import numpy as np

from dataset import get_data
import matplotlib.pyplot as plt


class SimpleLinearRegression:
    def __init__(self, learning_rate: float = 0.0001):
        self.weights = random.random()
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
        return np.mean(np.square(ground_truth - y_pred))

    def _gradient_descent(self, x_train, y_train, y_pred) -> None:
        """
        Given a loss vector, update the parameters of the model W.R.T the loss.
        """
        # delta L / delta B_0
        # partial_derivative_weights = -np.dot(x_train.T, y_train - y_pred) / len(x_train)
        # partial_derivative_bias = -np.sum(y_train - y_pred) / len(x_train)

        partial_derivative_bias = (-1 * (y_train - y_pred).sum()) / len(x_train)
        # delta L / delta B_1
        partial_derivative_weights = (-1 * x_train.dot(y_train - y_pred).sum()) / len(x_train)

        # Do gradient descent
        self.weights -= self.learning_rate * partial_derivative_weights
        self.bias -= self.learning_rate * partial_derivative_bias

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 5000) -> None:
        """

        Args:
            x_train: The training dataset (input features)
            y_train: The ground truth labels
            epochs: The number of epochs to train for

        Returns:
            Nothing. The weights of this model is fitted to the given dataset.
        """

        for epoch in range(epochs):
            # Perform batch gradient descent
            y_pred = self.predict(x_train)

            # Calculate l2 loss (Optional)
            loss = self.l2_loss(y_train, y_pred)

            # Do gradient descent
            self._gradient_descent(x_train, y_train, y_pred)

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
        pred = np.dot(X, self.weights) + self.bias
        return pred

    def visualize(self, X: np.ndarray, Y: np.ndarray, y_pred) -> None:
        plt.scatter(X, Y)
        plt.plot(X, y_pred)
        plt.show()


def do_linear_regression():
    # Step 1. Retrieve the data.
    x_train, y_train = get_data('Salary_Data.csv')

    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}')

    # Step 2. Define model
    # Before proceeding, make sure that SimpleLinearRegression is Defined
    model = SimpleLinearRegression(learning_rate=0.001)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)

    # Pred
    model.visualize(x_train, y_train, y_pred)
    print(f'Weight: {model.weights}, bias: {model.bias}')
    print(f'X: {x_train}')


if __name__ == '__main__':
    do_linear_regression()
