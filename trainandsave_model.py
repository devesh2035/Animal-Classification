import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.cnn_model import CNNModel
from src.preprocess import load_data

def save_model(cnn_model, model_path='scripts/model.npz'):
    np.savez(model_path, 
             conv_kernel=cnn_model.conv_kernel, 
             dense_weights=cnn_model.dense_weights, 
             dense_bias=cnn_model.dense_bias)
    print(f"Model saved to {model_path}")

def main():
    train_images, train_labels, _ = load_data('dataset/train')
    model = CNNModel()
    model.train_cnn(train_images, train_labels)
    save_model(model)

if __name__ == "__main__":
    main()
