import copy

import numpy as np
import pytest
import sys
import typing as t

# Adds higher directory to python modules path.
# Don't ever do this in production
# and don't put numbers as module names
sys.path.append('..')


from exercises import create_nd_array_with_values_1_to_n, create_none_vector_of_size_10, \
    create_random_images, element_wise_add, get_values_greater_than, min_max_norm_target_matrix, \
    transpose_bchw_to_bhwc, create_batch_of_ones_using_stack, replace_2d_border_with_zeros
from .conftest import assert_is_numpy_array


@pytest.mark.parametrize('first_array, second_array', [
    pytest.param(np.random.randint(1, 3, size=(3,)), np.random.randint(2, 234, size=(3,))),
    pytest.param(np.random.randint(-100, 300, size=(3, 5, 3, 10)), np.random.randint(-2, 54, size=(3, 5, 3, 10)))
])
def test_element_wise_add(first_array: np.ndarray, second_array: np.ndarray):
    added = element_wise_add(first_array, second_array)
    assert_is_numpy_array(added)


def test_create_null_vector_of_size_10():
    null_vec = create_none_vector_of_size_10()
    assert np.all(null_vec == None), f'Not all values in {null_vec} is None'


@pytest.mark.parametrize('numpy_array, greater_than_threshold, expected', [
    pytest.param(np.array([1, 2, 3, 4]), 2, np.array([3, 4])),
    pytest.param(np.array([1, 12, 19, 2, 3, 10]), 10, np.array([12, 19]))
])
def test_get_values_greater_than(numpy_array: np.ndarray,
                                 greater_than_threshold: int,
                                 expected: np.ndarray):
    actual = get_values_greater_than(numpy_array, greater_than_threshold)
    assert np.all(actual == expected), f'Expected: {expected}, Actual: {actual}'


@pytest.mark.parametrize('n', [
    1, 9, 200, 500
])
def test_create_nd_array_with_values_1_to_n(n: int):
    one_to_n_array = create_nd_array_with_values_1_to_n(n)
    assert_is_numpy_array(one_to_n_array)
    assert one_to_n_array.shape == (n,), f'Expected tuple with shape: {(n,)}, actual: {one_to_n_array}'
    for i in range(n):
        assert one_to_n_array[i] == i + 1, f'Expected: {i + 1}, Actual: {one_to_n_array[i]}'


def test_min_max_norm_target_matrix():
    matrix = np.random.randint(1, 100, size=(5, 5))
    actual = min_max_norm_target_matrix(matrix)
    row_axis = 0
    expected = (matrix - matrix.min(axis=row_axis)) / (matrix.max(axis=row_axis) - matrix.min(axis=row_axis))
    assert np.all(actual == expected)


@pytest.mark.parametrize('batch_size, channel, height, width', [
    pytest.param(3, 3, 16, 16),
    pytest.param(3, 10, 18, 9),
])
def test_create_random_images(batch_size: int, channel: int, height: int, width: int):
    batch_of_images = create_random_images(batch_size, channel, height, width)
    assert_is_numpy_array(batch_of_images)
    assert tuple(batch_of_images.shape) == (batch_size, channel, height, width)


def test_transpose_bchw_to_bhwc():
    images = transpose_bchw_to_bhwc()
    assert_is_numpy_array(images)
    assert np.all(images.shape == (4, 32, 16, 3))


@pytest.mark.parametrize('batch_size,shape_dim', [
    pytest.param(16, (3, 5, 4, 2)),
    pytest.param(16, (1,)),
    pytest.param(16, (100, 99, 98))
])
def test_create_batch_of_ones_using_stack(batch_size: int, shape_dim: t.Tuple[int, ...]):
    output = create_batch_of_ones_using_stack(batch_size, shape_dim)
    actual_shape = tuple(output.shape)
    expected_shape = ((batch_size, ) + shape_dim)
    assert actual_shape == expected_shape, f'Shape should be equal to: {expected_shape}. Got: {actual_shape}'
    assert np.all(output == 1), 'Values should all be equal to ones'


@pytest.mark.parametrize('original', [
    pytest.param(np.random.randint(1, 10, (4, 4))),
    pytest.param(np.random.randint(1, 10, (10, 2))),
    pytest.param(np.random.randint(1, 10, (10, 3))),
])
def test_replace_2d_border_with_zeros(original: np.ndarray):
    actual = copy.deepcopy(original)
    replace_2d_border_with_zeros(actual)

    # Create ground-truth using for loop
    for i in range(original.shape[0]):
        original[i, 0] = 0
        original[i, -1] = 0
    for j in range(original.shape[1]):
        original[0, j] = 0
        original[-1, j] = 0

    assert np.all(actual == original)
