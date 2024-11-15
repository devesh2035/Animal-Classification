# src/cnn_model.py
from src.preprocess import load_data
from src.convolution import conv2d
from src.activation import relu, sigmoid, relu_derivative, sigmoid_derivative
from src.pooling import max_pooling
from src.dense_layer import dense_layer
import numpy as np

class CNNModel:
    def __init__(self):
        self.conv_kernel = np.random.rand(3, 3, 3) 
        self.dense_weights = np.random.rand(128*128, 10)  
        self.dense_bias = np.random.rand(10)  

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
                
         
                conv_output = conv2d(image, self.conv_kernel)
                conv_relu_output = relu(conv_output)
                pool_output = max_pooling(conv_relu_output, (2, 2))
                flattened = pool_output.flatten()
                dense_output = dense_layer(flattened, self.dense_weights, self.dense_bias)
                predictions = sigmoid(dense_output)
                
                loss = -label * np.log(predictions) - (1 - label) * np.log(1 - predictions)
                error = predictions - label

                d_dense_output = error * sigmoid_derivative(dense_output)
                d_dense_weights = np.outer(flattened, d_dense_output)
                d_dense_bias = d_dense_output
                
                d_pool_output = d_dense_output.dot(self.dense_weights.T).reshape(pool_output.shape)
                
                # Propagate through the ReLU activation and max pooling
                d_conv_relu_output = np.zeros_like(conv_relu_output)
                for i in range(0, conv_relu_output.shape[0], 2):
                    for j in range(0, conv_relu_output.shape[1], 2):
                        max_idx = np.unravel_index(
                            np.argmax(conv_relu_output[i:i+2, j:j+2]), (2, 2)
                        )
                        d_conv_relu_output[i + max_idx[0], j + max_idx[1]] = d_pool_output[i // 2, j // 2]
                
                # Apply ReLU derivative
                d_conv_output = d_conv_relu_output * relu_derivative(conv_output)
                
                # Convolutional kernel gradient
                d_conv_kernel = np.zeros_like(self.conv_kernel)
                for i in range(d_conv_output.shape[0]):
                    for j in range(d_conv_output.shape[1]):
                        d_conv_kernel += image[i:i+3, j:j+3] * d_conv_output[i, j]
                
    
                self.dense_weights -= lr * d_dense_weights
                self.dense_bias -= lr * d_dense_bias
                self.conv_kernel -= lr * d_conv_kernel


    def evaluate(self, test_data):
        correct_predictions = 0
        total_samples = len(test_data)
        
        for image, label in test_data:
            predictions = self.build_and_run_cnn(image)
            predicted_label = np.argmax(predictions)  
            
          
            if predicted_label == np.argmax(label):  
                correct_predictions += 1
        
        accuracy = correct_predictions / total_samples
        return accuracy   
       

