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


def check_nb_of_data_in_dataset(dataset: tf.data.Dataset):
    """
    check the number of data in the dataset

    :param dataset: tf.data.Dataset
    :return:
    """
    nb_of_batches = dataset.cardinality().numpy()
    nb_of_data = nb_of_batches * BATCH_SIZE
    print(f"Nb of data: {nb_of_data} | Nb of batches: {nb_of_batches}")
    return None


def check_nb_of_classes_in_dataset(dataset: tf.data.Dataset):
    """
    check the number of classes in the dataset

    :param dataset: tf.data.Dataset
    :return:
    """
    class_names = dataset.class_names
    print(f"Nb of classes: {len(class_names)} | Class names: {class_names}")
    return None


def load_split_dataset(val_split: float, test_split: float, silent_console: bool = True):
    """
    load and split the dataset

    :return: tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
    """
    # get training dataset
    eval_split = val_split + test_split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=eval_split,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # get data to eval (validation and test)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=eval_split,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # # split validation dataset into validation and test datasets
    # val_split_in_eval = val_split / eval_split
    # test_split_in_eval = test_split / eval_split
    # if val_split_in_eval + test_split_in_eval != 1:
    #     raise ValueError("The sum of val_split_in_eval and test_split_in_eval must be equal to 1")
    # val_ds = validation_dataset.take(int(val_split_in_eval * len(validation_dataset)))
    # test_ds = validation_dataset.skip(int(test_split * len(validation_dataset)))

    if not silent_console:
        print("--- Train dataset: ")
        check_nb_of_data_in_dataset(train_ds)
        check_nb_of_classes_in_dataset(train_ds)
        print("--- Validation dataset: ")
        check_nb_of_data_in_dataset(val_ds)
        check_nb_of_classes_in_dataset(val_ds)
        # print("--- Test dataset: ")
        # check_nb_of_data_in_dataset(test_ds)
        # check_nb_of_classes_in_dataset(test_ds)

    return train_ds, val_ds


def check_dataset_classes(dataset: tf.data.Dataset, dataset_type: str):
    """
    check the dataset classes

    :return:
    """
    class_names = dataset.class_names
    print(f"Dataset: {dataset_type} | Nb of classes: {len(class_names)} | Class names: {class_names}")


def check_batch_size(dataset: tf.data.Dataset, dataset_type: str):
    """
    check the batch size of the dataset

    :return:
    """
    print(f"Checking batch size of the {dataset_type} dataset...")
    for image_batch, labels_batch in dataset:
        print(f"Image batch shape: {image_batch.shape}")
        print(f"Label batch shape: {labels_batch.shape}")
        break


def display_img_sample_of_dataset(dataset: tf.data.Dataset, dataset_type: str):
    """
    display a sample of images from the dataset

    :return:
    """
    plt.figure(figsize=(10, 10))
    class_names = dataset.class_names

    # Take one batch of images
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)  # Create the subplot
            ax.imshow(images[i].numpy().astype("uint8"))  # Show the image

            # Set the title on the subplot (not the entire plot)
            ax.set_title(f"{class_names[labels[i]]}", fontsize=8)
            ax.axis("off")  # Remove the axis labels

    # Adjust layout to prevent overlapping titles
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.3)

    # Add a title to the entire plot
    plt.suptitle(f"Sample of images from the {dataset_type} dataset", fontsize=16)
    plt.show()


if __name__ == '__main__':
    show_const_var_settings()

    # Check dataset infos
    check_data_dir()
    nb_img_data = get_dataset_info(data_dir)
    print(f"# {dataset_name.upper()} dataset contains {nb_img_data} images")

    # Load split dataset
    train_dataset, val_dataset = load_split_dataset(0.2, 0.2, silent_console=True)

    # Check dataset classes
    check_dataset_classes(train_dataset, "train")
    check_dataset_classes(val_dataset, "validation")

    # Display sample of images
    check_batch_size(train_dataset, "train")
    display_img_sample_of_dataset(train_dataset, "train")

    # Display sample of images
    check_batch_size(val_dataset, "validation")
    display_img_sample_of_dataset(val_dataset, "validation")





