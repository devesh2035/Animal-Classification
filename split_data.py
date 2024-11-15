import os
import shutil
import random

def split_dataset(image_dir, train_dir, test_dir, test_size=0.2):
    # Create directories for train and test if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all images in the dataset
    images = os.listdir(image_dir)
    random.shuffle(images)

    # Calculate the split index
    split_idx = int(len(images) * (1 - test_size))

    # Split images into train and test
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Copy images to respective folders
    for img in train_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(test_dir, img))

    print(f"Total images: {len(images)}")
    print(f"Training images: {len(train_images)}")
    print(f"Testing images: {len(test_images)}")

if __name__ == "__main__":
    image_dir = "../dataset/images"  
    train_dir = "../dataset/train"  
    test_dir = "../dataset/test"    
    split_dataset(image_dir , train_dir , test_dir , test_size= 0.2)