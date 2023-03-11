import typing as t
import numpy as np


def element_wise_add(np_array: np.ndarray, second_numpy_array: np.ndarray) -> np.ndarray:
    """

    Args:
        np_array: The left hand side numpy array
        second_numpy_array: The right hand side numpy array

    Returns:
        A numpy array where each element is the sum of the element at index i, respectively.

    E.g. A = np.array([1, 2]), B = np.array([3, 4])
         Output = A + B =  np.array([4, 6])
    """
    raise NotImplementedError


def create_none_vector_of_size_10() -> np.ndarray:
    """
    Returns: A numpy array of size (10, ) with all values None
    """
    raise NotImplementedError


def create_nd_array_with_values_1_to_n(n: int) -> np.ndarray:
    """
    Create a numpy array with the following values
    [1, 2, 3, ... , n]
    Where n = the number of elements in an array
    """
    raise NotImplementedError


def get_values_greater_than(numpy_array: np.ndarray, greater_than_threshold: int) -> np.ndarray:
    """
    Given a numpy array and an integer, return a new numpy array
    with values greater than the value
    Args:
        numpy_array: A numpy array
        greater_than_threshold: An integer value. This will act as a threshold value, determining what
        values will be included in the output array

    Returns:
        A numpy array with the output values that are greater than the specified "greater_than_threshold" value.

    E.g.
        A = np.array([1, 2, 3, 4]), greater_than_threshold = 2
        output = np.array([3, 4])
    """
    raise NotImplementedError


def min_max_norm_target_matrix(matrix: np.ndarray):
    """
    Apply min max normalization to the matrix on a column by column basis.
    For information on what min-max normalization is, check out
    https://www.oreilly.com/library/view/hands-on-machine-learning/9781788393485/fd5b8a44-e9d3-4c19-bebb-c2fa5a5ebfee.xhtml
    Or look up google
    Args:
        matrix: The target matrix to normalize

    Returns:
        The min max normalized matrix

    e.g. Input: [
            [1, 12],
            [2, 7],
            [3, 9],
            [4, 16],
         ]

         Output: [
            [0, 0.55555556]
            [0.33333333, 0]
            [0.66666667, 0.22222222]
            [1, 1]
         ]

    """
    raise NotImplementedError


def create_random_images(batch_size, channel, height, width) -> np.ndarray:
    """
    Create a random numpy array that is of shape
    (batch x channel x height x width)
    Returns:
        np.ndarray
    """
    raise NotImplementedError


def transpose_bchw_to_bhwc() -> np.ndarray:
    """
    Given an image of size
    (batch x channel x height x width),
    transpose the image so that it is of shape:
    (batch x height x width x channel)

    Returns: The transposed image
    """
    batch_size, channel, height, width = 4, 3, 32, 16
    image = np.random.randn(batch_size, channel, height, width)
    # TODO: Transpose this image so that it is of shape
    # (batch_size, height, width, channel)
    # IMPORTANT: DO NOT USE reshape
    raise NotImplementedError


def create_batch_of_ones_using_stack(batch_size: int,
                                     shape_dim: t.Tuple[int, ...]) -> np.ndarray:
    """
    Given a batch_size and a shape dimension,
    Args:
        batch_size: The batch_size of the int
        shape_dim: A tuple notating the shape of each batch of data

    Returns:
        A numpy array of ones with shape (batch_size, *shape_dim).

        e.g. batch_size = 16, shape_dim = (3, 2, 2,)
        then result should be of shape (16, 3, 2, 2)
    """
    results = []
    for i in range(batch_size):
        # TODO: update me
        raise NotImplementedError
    # modify results here into a numpy array using np.stack()
    # Output shape should be of shape(batch_size, *shape_dim).
    # e.g. batch_size = 16, shape_dim = (3, 2, 2,)
    # then result should be of shape (16, 3, 2, 2)
    return results


def replace_2d_border_with_zeros(np_array: np.ndarray) -> None:
    """
    Given a numpy array of more than one axis, replace borders with zeros.
    Important: do this without using a For / while loop.
    Args:
        np_array: A 2 dimensional array

    Returns:
        Nothing. The np_array must be modified IN-PLACE
    """
    assert len(np_array.shape) == 2, f'np_array must be 2-dimensional. Shape of np_array: {np_array.shape}'
    raise NotImplementedError
