import numpy as np
import pandas as pd


def get_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    # now, we need to split the dataset into x_train, y_train
    # y_train contains the ground_truth labels
    # x_train will be our features

    # first column contains years of experience. We will attempt to map years of experience
    # to the second column, which is the salary.
    x_train = df.iloc[:, 0]

    # DataFrame containing the corresponding salaries
    y_train = df.iloc[:, 1]
    return x_train.to_numpy().reshape(-1), y_train.to_numpy().reshape(-1)
