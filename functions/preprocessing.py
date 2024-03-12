import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

RSEED = 42

def test_2(number):
    return number + 1

def df_from_directory(folder_path):
    '''
    Input: Path to parent folder that includes subfolders where images are stored. The names of the subfolders are used as classes.
    Return: df that stores the path to all images ('image_path') and their classes/folder names ('class')
    '''
    data = []

    # Iterate through each class folder
    for folder_class in os.listdir(folder_path):
        class_path = os.path.join(folder_path, folder_class)
    
        # Iterate through each image in the class folder
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            data.append({'image_path': image_path, 'class': folder_class})

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    return df


def rename_subfolder(parent_dir, subfolder_name, new_name):
    subfolder_path = os.path.join(parent_dir, subfolder_name)
    new_path = os.path.join(parent_dir, new_name)

    # Check if the subfolder exists
    if os.path.exists(subfolder_path):
        # Rename the subfolder
        os.rename(subfolder_path, new_path)
        print(f"{subfolder_name} renamed to {new_name} successfully.")
    else:
        print(f"Subfolder {subfolder_name} doesn't exist in {parent_dir}.")
    return new_name


def img_dataset_from_dir_and_split_train_val(data_path):
    ''' 
    Input: Path to parent directory where the images are stored in subfolders/classes
    Return training and validation data set at 80:20 ratio. Crops and resizes images to 224x224px. RSEED = 42
    '''
    datasets = tf.keras.utils.image_dataset_from_directory(
        data_path, 
        validation_split = 0.2,
        subset = "both", 
        seed = RSEED,
        image_size = (224, 224),
        crop_to_aspect_ratio = True,
        label_mode = 'categorical'
    )
    return datasets


def plot_color_distribution(image_path):
    image = Image.open(image_path)
    colors = ['Red', 'Green', 'Blue']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create subplots with 1 row and 3 columns
    for i, color in enumerate(colors):
        axes[i].hist(np.array(image)[:, :, i].flatten(), bins=256, color=color.lower(), alpha=0.5, label=color)
        axes[i].set_title(f'{color} Channel')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    return plt.show()