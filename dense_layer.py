import numpy as np

def dense_layer(input_data, weights, bias):
    return np.dot(input_data, weights) + bias
