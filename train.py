import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from cnn_model import CNNModel  # Import the CNNModel class
from preprocess import load_data  # Import load_data function if defined elsewhere

def main():
    train_images, train_labels, _ = load_data('dataset/train')
    model = CNNModel()  # Create an instance of CNNModel
    model.train_cnn(train_images, train_labels)  # Call train_cnn on the model instance

if __name__ == "__main__":
    main()
