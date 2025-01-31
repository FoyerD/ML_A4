import os
import scipy.io
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Path to your data
image_dir = "./code_files/ML_A4/data_formaters/jpg"  # Folder containing images
mat_file = "./code_files/ML_A4/data_formaters/imagelabels.mat"  # Path to your .mat file

# Load .mat file
mat_data = scipy.io.loadmat(mat_file)
labels = mat_data['labels'][0]  # Extract labels (check if this key matches your .mat file)

# Create base directory for dataset
base_dir = './code_files/ML_A4/data_formaters/dataset'

image_filenames = os.listdir(image_dir)


# Split the data (50-25-25 split)
X_train, X_temp, y_train, y_temp = train_test_split(image_filenames, labels, test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Function to prepare dataset
def prepare_dataset(X, y, split):
    # Make split dir
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir)

    # Make classes dir
    for label in np.unique(labels):
         os.makedirs(os.path.join(split_dir, str(label)))
    
    for image_filename, label in zip(X, y):
        # Image path
        src_image_path = os.path.join(image_dir, image_filename)  # Adjust path if needed
        dst_image_path = os.path.join(split_dir, str(label), image_filename)
        shutil.copy(src_image_path, dst_image_path)

# Prepare train, validation, and test datasets
prepare_dataset(X_train, y_train, 'train')
prepare_dataset(X_val, y_val, 'valid')
prepare_dataset(X_test, y_test, 'test')
