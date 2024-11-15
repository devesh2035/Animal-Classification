# src/preprocess.py
import os
import numpy as np
from PIL import Image

def load_data(data_dir, target_size=(128, 128)):
    """
    Load images from a directory and resize them to the target size.
    """
    print(f"Loading data from directory: {data_dir}")
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir)) 
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue 
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img, dtype=np.float32) / 255.0 
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return np.array(images), np.array(labels), class_names

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess a single image for prediction.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def load_and_preprocess_data(train_dir, test_dir, target_size=(128, 128)):
    """
    Load and preprocess data for training and testing.
    """
    print("Loading and preprocessing training data...")

    train_images, train_labels, train_class_names = load_data(train_dir, target_size)
    
    
    print("Loading and preprocessing test data...")
    test_images, test_labels, test_class_names = load_data(test_dir, target_size)

    return (train_images, train_labels), (test_images, test_labels), train_class_names

# Main function to test the data loading and preprocessing
def main(train_dir, test_dir):
    """
    Main function to load and preprocess the data.
    Accepts paths for training and testing directories as arguments.
    """
    # Check if the dataset directories exist
    if not os.path.exists(train_dir):
        print(f"Train data directory not found: {train_dir}")
        return

    if not os.path.exists(test_dir):
        print(f"Test data directory not found: {test_dir}")
        return

    print("Testing data loading and preprocessing...")
    (train_images, train_labels), (test_images, test_labels), class_names = load_and_preprocess_data(train_dir, test_dir)

    print(f"Preprocessing complete. Found {len(class_names)} classes.")
    print(f"Training images shape: {train_images.shape}, Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}")

if __name__ == "__main__":
    # Defining paths for training and testing datasets
    train_data_dir = os.path.join('dataset', 'train')
    test_data_dir = os.path.join('dataset', 'test')
    
    # Call the main function with the specified data directories
    main(train_data_dir, test_data_dir)
