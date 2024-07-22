import os
BACKEND = 'jax'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras import random
from keras_efficient_kan import KANLinear


def generate_random_tensor(shape):
    return random.normal(shape=shape, dtype=backend.floatx())

@pytest.fixture(scope="module")
def kan_layer_2d():
    return KANLinear(units=5, grid_size=3, spline_order=3)

@pytest.fixture(scope="module")
def kan_layer_3d():
    return KANLinear(units=6, grid_size=4, spline_order=2)

@pytest.fixture(scope="module")
def kan_layer_4d():
    return KANLinear(units=4, grid_size=3, spline_order=3)

@pytest.fixture(scope="module")
def kan_layer_5d():
    return KANLinear(units=4, grid_size=3, spline_order=3)

def test_kanlinear_2d(kan_layer_2d):
    assert keras.backend.backend() == BACKEND
    batch_size, in_features = 32, 10
    input_2d = generate_random_tensor((batch_size, in_features))
    output_2d = kan_layer_2d(input_2d)
    assert output_2d.shape == (batch_size, 5), f"Expected shape {(batch_size, 5)}, but got {output_2d.shape}"

def test_kanlinear_3d(kan_layer_3d):
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, in_features = 16, 20, 8
    input_3d = generate_random_tensor((batch_size, time_steps, in_features))
    output_3d = kan_layer_3d(input_3d)
    assert output_3d.shape == (batch_size, time_steps, 6), f"Expected shape {(batch_size, time_steps, 6)}, but got {output_3d.shape}"

    test_input = generate_random_tensor((batch_size, 1, in_features))
    repeated_input = ops.tile(test_input, [1, time_steps, 1])
    repeated_output = kan_layer_3d(repeated_input)

    first_timestep_output = repeated_output[:, 0, :]
    for t in range(1, time_steps):
        assert ops.all(ops.abs(repeated_output[:, t, :] - first_timestep_output) < 1e-6), f"Transformation is not consistent for identical inputs at step {t}"

def test_kanlinear_4d(kan_layer_4d):
    assert keras.backend.backend() == BACKEND
    batch_size, dim1, dim2, in_features = 8, 10, 15, 6
    input_4d = generate_random_tensor((batch_size, dim1, dim2, in_features))
    output_4d = kan_layer_4d(input_4d)
    expected_shape = (batch_size, dim1, dim2, 4)
    assert output_4d.shape == expected_shape, f"Expected shape {expected_shape}, but got {output_4d.shape}"

    test_input = generate_random_tensor((1, 1, 1, in_features))
    repeated_input = ops.tile(test_input, [batch_size, dim1, dim2, 1])
    repeated_output = kan_layer_4d(repeated_input)

    first_output = repeated_output[0, 0, 0, :]
    assert ops.all(ops.abs(repeated_output - first_output) < 1e-6), "Transformation is not consistent for identical inputs across 4D"

def test_kanlinear_5d(kan_layer_5d):
    assert keras.backend.backend() == BACKEND
    batch_size, dim1, dim2, dim3, in_features = 3, 4, 5, 2, 6

    # Use Keras to generate random tensor
    input_5d = random.normal(shape=(batch_size, dim1, dim2, dim3, in_features))

    # Create a slice with the same values
    same_slice = random.normal(shape=(1, 1, dim3, in_features))

    # Insert the same_slice into specific positions
    if BACKEND == 'jax':
        input_5d = input_5d.at[0, 1, 0].set(same_slice[0, 0])
        input_5d = input_5d.at[1, 0, 2].set(same_slice[0, 0])
        input_5d = input_5d.at[2, 1, 0].set(same_slice[0, 0])
    else:
        input_5d = input_5d.numpy()  # Convert to numpy for easy assignment
        input_5d[0, 1, 0] = same_slice[0, 0]
        input_5d[1, 0, 2] = same_slice[0, 0]
        input_5d[2, 1, 0] = same_slice[0, 0]
        input_5d = ops.convert_to_tensor(input_5d)  # Convert back to tensor
        
    output_5d = kan_layer_5d(input_5d)
    expected_shape = (batch_size, dim1, dim2, dim3, 4)
    assert output_5d.shape == expected_shape, f"Expected shape {expected_shape}, but got {output_5d.shape}"

    slices_to_check = [
        output_5d[0, 1, 0],
        output_5d[1, 0, 2],
        output_5d[2, 1, 0]
    ]

    for i in range(1, len(slices_to_check)):
        assert ops.all(ops.abs(slices_to_check[0] - slices_to_check[i]) < 1e-5), \
            f"Transformation is not consistent for identical inputs at slice {i}"

    different_slice = output_5d[0, 1, 1]
    assert not ops.all(ops.abs(slices_to_check[0] - different_slice) < 1e-5), \
        "Transformation is incorrectly consistent for different inputs"
