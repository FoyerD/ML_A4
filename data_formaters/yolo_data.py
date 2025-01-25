import os
import scipy.io
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Path to your data
image_dir = "jpg"  # Folder containing images
mat_file = "imagelabels.mat"  # Path to your .mat file

# Load .mat file
mat_data = scipy.io.loadmat(mat_file)
labels = mat_data['labels'][0]  # Extract labels (check if this key matches your .mat file)

# Create base directory for dataset
base_dir = 'datasets'
image_filenames = os.listdir(image_dir)


# Split the data (50-25-25 split)
X_train, X_temp, y_train, y_temp = train_test_split(image_filenames, labels, test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)


# Create necessary directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# Function to write labels in YOLO format
def write_yolo_label(label_path, class_id):
    with open(label_path, 'w') as f:
        f.write(f"{class_id}\n")  # Only one label per image for classification

# Function to prepare dataset
def prepare_dataset(X, y, split):
    for image_filename, label_class in zip(X, y):
        # Image path
        src_image_path = os.path.join(image_dir, image_filename)  # Adjust path if needed
        dst_image_path = os.path.join(base_dir, split, 'images', image_filename)
        shutil.copy(src_image_path, dst_image_path)

        label_filename = image_filename.replace('.jpg', '.txt')
        # Create YOLO label file
        label_filepath = os.path.join(base_dir, split, 'labels', label_filename)
        write_yolo_label(label_filepath, label_class)

# Prepare train, validation, and test datasets
prepare_dataset(X_train, y_train, 'train')
prepare_dataset(X_val, y_val, 'val')
prepare_dataset(X_test, y_test, 'test')
