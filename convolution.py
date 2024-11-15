import numpy as np

def conv2d(image, kernel):
    output_shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1)
    conv_output = np.zeros(output_shape)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            conv_output[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return conv_output
