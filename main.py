# scripts/main.py
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cnn_model import CNNModel
from src.preprocess import load_and_preprocess_data
from src.trainandsave_model import main
from src.predict import predict_image

def main():
  
    train_data_dir = os.path.join('dataset', 'train')
    test_data_dir = os.path.join('dataset', 'test')

 
    print("Loading and preprocessing dataset...")
    train_data, test_data = load_and_preprocess_data(train_data_dir, test_data_dir)

 
    print("Initializing CNN model...")
    cnn_model = CNNModel()

  
    print("Training the model...")
    cnn_model.train_cnn(*train_data) 

  
    print("Do you want to predict or evaluate? (Type 'predict' for prediction, anything else for evaluation)")
    choice = input().strip().lower()

    if choice == 'predict':
        print("Enter the image path for prediction:")
        image_path = input().strip()
        result = predict_image(cnn_model, image_path)
        print(f"Predicted animal class: {result}")
    else:
        print("Evaluating the model on test data...")
        accuracy = cnn_model.evaluate(test_data)
        print(f"Model accuracy on test data: {accuracy:.2f}%")

if __name__ == "__main__":
    main()