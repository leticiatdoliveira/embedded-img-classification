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
    """
    print the const settings

    :param:
    :return:
    """
    print("Printing const settings...")
    print("\t data_dir: " + data_dir)


def check_data_dir():
    """
    check if the data_dir exists

    :param:
    :return:
    """
    if os.path.exists(data_dir):
        print("Data_dir found !")
    else:
        print("The data_dir not found !")


def get_dataset_info(directory: str) -> int:
    """
    get the number of images in the dataset

    :param directory: str
    :return: int
    """
    dir_path = Path(directory)
    image_count = len(list(dir_path.glob('*/*.jpg')))
    image_count += len(list(dir_path.glob('*/*.JPG')))
    return image_count


def split_dataset(dataset: tf.data.Dataset, train_split: float, val_split: float, test_split: float, shuffle_size: int,
                  shuffle: bool = True):
    """
    split dataset into train, validation and test datasets

    :param dataset: tf.data.Dataset
    :param train_split: float
    :param val_split:
    :param test_split:
    :param shuffle: bool
    :param shuffle_size: int
    :return:
    """
    assert (train_split + test_split + val_split) == 1

    dataset_size = dataset.cardinality().numpy()

    # shuffle data if needed
    if shuffle:
        ds = dataset.shuffle(shuffle_size, seed=123)

    # set each dataset size
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    # partition the dataset
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size).skip(val_size)

    return train_dataset, val_dataset, test_dataset


def load_split_dataset():
    """
    load and split the dataset

    :return: tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=default_image_size,
        batch_size=BATCH_SIZE,
    )

    training_dataset, validation_dataset, test_dataset = split_dataset(dataset,
                                                                       0.7,
                                                                       0.15,
                                                                       0.15,
                                                                       1000,
                                                                       shuffle=True)

    return training_dataset, validation_dataset, test_dataset


if __name__ == '__main__':
    show_const_var_settings()

    # Check dataset infos
    check_data_dir()
    nb_img_data = get_dataset_info(data_dir)
    print(f"# {dataset_name.upper()} dataset contains {nb_img_data} images")

    # Load split dataset
    train_dataset, val_dataset, test_dataset = load_split_dataset()

    # Show dataset cardinality
    print(f"train_dataset: {train_dataset.cardinality().numpy()}")
    print(f"val_dataset: {val_dataset.cardinality().numpy()}")
    print(f"test_dataset: {test_dataset.cardinality().numpy()}")
