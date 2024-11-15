import numpy as np

def max_pooling(image, pool_size=(2, 2)):
    pool_height, pool_width = pool_size
    output_shape = (image.shape[0] // pool_height, image.shape[1] // pool_width)
    pooled_output = np.zeros(output_shape)
    for i in range(0, image.shape[0], pool_height):
        for j in range(0, image.shape[1], pool_width):
            pooled_output[i // pool_height, j // pool_width] = np.max(image[i:i+pool_height, j:j+pool_width])
    return pooled_output