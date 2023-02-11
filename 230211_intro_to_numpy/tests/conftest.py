import numpy as np
import pytest


def assert_is_numpy_array(input_data):
    if input_data is None:
        raise TypeError('Input data should not be "None"')
    assert isinstance(input_data, np.ndarray), 'Return type must be a numpy array'


@pytest.fixture
def random_test_image_dimensions():
    return [
        pytest.param(3, 3, 16, 16),
        pytest.param(3, 10, 18, 9),
    ]