import sys
import os
import numpy as np
from cnn_model import CNNModel
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def predict_image(image_path, model_path='model.npz'):

    model_data = np.load(model_path)
    conv_kernel = model_data['conv_kernel']
    dense_weights = model_data['dense_weights']
    dense_bias = model_data['dense_bias']

  
    model = CNNModel()
    model.conv_kernel = conv_kernel
    model.dense_weights = dense_weights
    model.dense_bias = dense_bias


    processed_image = preprocess_image(image_path)
    
 
    output = model.build_and_run_cnn(processed_image)
    return np.argmax(output)
    
def preprocess_image(image_path):
 
    image = Image.open(image_path).convert('L')  
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return image_array

# Example usage
if __name__ == "__main__":
    image_path = 'kitty.jpg'
    prediction = predict_image(image_path)
    print(f'Predicted class: {prediction}')
