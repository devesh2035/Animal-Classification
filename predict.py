import sys
import os
import numpy as np
from cnn_model import CNNModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def predict_image(image, model_path='results/model_output/model.npz'):
    # Load trained model
    model_data = np.load(model_path)
    conv_kernel = model_data['conv_kernel']
    dense_weights = model_data['dense_weights']
    dense_bias = model_data['dense_bias']

    # Initialize CNNModel with loaded weights and biases
    model = CNNModel()
    model.conv_kernel = conv_kernel
    model.dense_weights = dense_weights
    model.dense_bias = dense_bias

    # Preprocess the input image (resize, normalize)
    processed_image = preprocess_image(image)
    
    # Predict using the trained model
    output = model.build_and_run_cnn(processed_image)
    return np.argmax(output)
    
def preprocess_image(image):
    # Implement preprocessing: resize and normalize
    return image

# Example usage
if __name__ == "__main__":
    # Replace with actual image and model path
    image = np.random.rand(128, 128, 3)  # Dummy image
    prediction = predict_image(image)
    print(f'Predicted class: {prediction}')
