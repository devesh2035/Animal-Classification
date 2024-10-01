# src/cnn_model.py
from preprocess import load_data
from convolution import conv2d
from activation import relu, sigmoid
from pooling import max_pooling
from dense_layer import dense_layer
import numpy as np

class CNNModel:
    def __init__(self):
        # Initialize weights and biases
        self.conv_kernel = np.random.rand(3, 3, 3)  # Example kernel
        self.dense_weights = np.random.rand(128*128, 10)  # Example weights for dense layer
        self.dense_bias = np.random.rand(10)  # Example biases for dense layer

    def build_and_run_cnn(self, image):
        conv_output = conv2d(image, self.conv_kernel)
        conv_output = relu(conv_output)
        pool_output = max_pooling(conv_output, (2, 2))
        flattened = pool_output.flatten()
        dense_output = dense_layer(flattened, self.dense_weights, self.dense_bias)
        predictions = sigmoid(dense_output)
        return predictions

    def train_cnn(self, train_images, train_labels, epochs=10, lr=0.01):
        num_samples = len(train_images)
        for epoch in range(epochs):
            for i in range(num_samples):
                image = train_images[i]
                label = train_labels[i]
                
                # Forward pass
                conv_output = conv2d(image, self.conv_kernel)
                conv_output = relu(conv_output)
                pool_output = max_pooling(conv_output, (2, 2))
                flattened = pool_output.flatten()
                dense_output = dense_layer(flattened, self.dense_weights, self.dense_bias)
                predictions = sigmoid(dense_output)
                
                # Compute loss and gradients (Dummy example)
                loss = -label * np.log(predictions) - (1 - label) * np.log(1 - predictions)
                error = predictions - label
                
                # Backward pass (simplified, for demonstration)
                # Update weights and biases
                self.dense_weights -= lr * np.outer(flattened, error)
                self.dense_bias -= lr * error

                print(f"Epoch {epoch+1}, Loss: {loss}")

    def evaluate(self, test_data):
        # Dummy evaluation method
        return 0.0  # Replace with actual evaluation logic

