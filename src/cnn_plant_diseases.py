import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf


# TF settings
AUTOTUNE = tf.data.AUTOTUNE

# Model hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 100

# Image settings param.
IMAGE_SIZE = 256
default_image_size = tuple((IMAGE_SIZE, IMAGE_SIZE))
image_size = 0
CHANNELS = 3

# Data settings
dataset_name = 'plant-village'
user_dir = '/Users/tempone/'
project_dir = 'Git/IC_vant/embedded-img-classification/'
data_dir: str = user_dir + project_dir + 'data/' + dataset_name + '/'


def show_const_var_settings():
    print("Printing const settings...")
    print("\t data_dir: " + data_dir)


def check_data_dir():
    if os.path.exists(data_dir):
        print("Data_dir found !")
    else:
        print("The data_dir not found !")


def get_dataset_info(dir: str) -> int:
    dir_path = Path(dir)
    image_count = len(list(dir_path.glob('*/*.jpg')))
    image_count += len(list(dir_path.glob('*/*.JPG')))
    return image_count


if __name__ == '__main__':
    show_const_var_settings()

    # Check dataset infos
    check_data_dir()
    nb_img_data = get_dataset_info(data_dir)
    print(f"# {dataset_name.upper()} dataset contains {nb_img_data} images")
