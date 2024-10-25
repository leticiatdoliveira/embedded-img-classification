import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_model_optimization as tfmot

# Model hyperparameters
BATCH_SIZE = 16
EPOCHS = 15

# Image settings param.
IMAGE_SIZE = 256

# Data settings
dataset_name = 'plant-village'
user_dir = '/Users/tempone/'
project_dir = 'Git/IC_vant/embedded-img-classification/'
data_dir: str = user_dir + project_dir + 'data/' + dataset_name + '/'

# Results settings
results_dir = user_dir + project_dir + 'results/' + dataset_name + '/'


def show_const_var_settings():
    """
    print the const settings

    :param:
    :return:
    """
    print("Printing const settings...")
    print("\t data_dir: " + data_dir)


def check_data_dir(silent_console: bool = True):
    """
    check if the data_dir exists

    :param:
    :return:
    """
    if os.path.exists(data_dir):
        print("Data_dir found !") if not silent_console else None
        return True
    else:
        print("The data_dir not found !") if not silent_console else None
        return False


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


def get_dataset_classes(dataset: tf.data.Dataset, dataset_type: str, silent_console: bool = True):
    """
    check the dataset classes

    :return:
    """
    class_names = dataset.class_names
    if not silent_console:
        print(f"Dataset: {dataset_type} | Nb of classes: {len(class_names)} | Class names: {class_names}")
    return class_names


def check_batch_size(dataset: tf.data.Dataset, dataset_type: str):
    """
    check the batch size of the dataset

    :return:
    """
    print(f"\n------ Checking batch size of the {dataset_type} dataset...")
    for image_batch, labels_batch in dataset:
        print(f"Image batch shape: {image_batch.shape}")
        print(f"Label batch shape: {labels_batch.shape}\n")
        break


def display_img_sample_of_dataset(dataset: tf.data.Dataset, dataset_type: str):
    """
    display a sample of images from the dataset

    :return:
    """
    plt.figure(figsize=(10, 10))
    class_names = dataset.class_names

    # Take one batch of images and create a subplot
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

    # Save the plot
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + f"sample_images_{dataset_type}.png")


def set_prefetch(dataset: tf.data.Dataset):
    """
    set the prefetch for the dataset

    :param dataset: tf.data.Dataset
    :return: tf.data.Dataset
    """
    AUTOTUNE = tf.data.AUTOTUNE

    return dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)


def normalize_dataset(dataset: tf.data.Dataset, silent_console: bool = True):
    """
    Normalize image data.
    RGB values are in the [0, 255] range, so we need to scale them to the [0, 1] range.

    :return:
    """
    print("Normalizing dataset...")
    normalizer_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = dataset.map(lambda x, y: (normalizer_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    # get the first image to check the pixel values
    if not silent_console:
        first_image = image_batch[0]
        print(f"First image pixel values -> Min: {np.min(first_image)} | Max: {np.max(first_image)} | "
              f"Shape: {first_image.shape}")
    return normalized_ds


def create_model(class_names: list, img_height: int, img_width: int):
    """
    Create a CNN model to classify image

    :return:
    """
    num_classes = len(class_names)
    image_shape = (img_height, img_width, 3)
    model = Sequential([
        layers.Input(shape=image_shape),
        layers.Resizing(img_height, img_width),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2), # Dropout layer to reduce overfitting
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])

    return model


def compile_mode(model: tf.keras.Model):
    """
    Compile the model

    :return:
    """
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.summary()

    return model


def train_model(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, nb_epochs: int = EPOCHS):
    """
    Train the model

    :return:
    """
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=nb_epochs
    )

    return hist


def show_model_fit(hist: tf.keras.callbacks.History, nb_epochs: int):
    """
    Show the model fit

    :return:
    """
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs_range = range(nb_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Save the plot
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + "model_fit.png")


def save_model(model: tf.keras.Model, model_name: str, file_format: str = "keras"):
    """
    Save the model

    :return:
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_file = model_name + "." + file_format
    model_path = results_dir + model_file
    model.save(model_path)


def show_first_data_in_dataset(dataset: tf.data.Dataset, class_names: list):
    """
    Show the first data in the dataset

    :return:
    """
    print("\n---- Showing the first data in the dataset...")
    # get the first batch of data
    for image, label in dataset.take(1):
        # Show the first image and label
        first_img = image[0]
        first_label = label[0]
        print(f"First image shape: {first_img.shape} | First label: {first_label}")
        print(f"First image pixel values -> Min: {np.min(first_img)} | Max: {np.max(first_img)}")

        # Display the first image
        plt.figure()
        plt.imshow(first_img)
        plt.title(f"First image example | Label: {first_label} | Class name: {class_names[first_label]}")
        plt.grid(False)
        plt.show()


if __name__ == '__main__':
    show_const_var_settings()

    # Check dataset infos
    if not check_data_dir():
        raise ValueError("Data dir not found !")
    nb_img_data = get_dataset_info(data_dir)
    print(f"# {dataset_name.upper()} dataset contains {nb_img_data} images\n")

    # Load split dataset
    train_dataset, val_dataset = load_split_dataset(0.2, 0.2, silent_console=True)

    # Check dataset classes
    train_classes = get_dataset_classes(train_dataset, "train")
    val_classes = get_dataset_classes(val_dataset, "validation")
    if train_classes != val_classes:
        raise ValueError("The classes in the train and validation datasets are different")
    else:
        print(f"Nb of class: {len(train_classes)} | Classes: {train_classes}\n")

    # Display sample of images
    check_batch_size(train_dataset, "train")
    display_img_sample_of_dataset(train_dataset, "train")

    # Display sample of images
    check_batch_size(val_dataset, "validation")
    display_img_sample_of_dataset(val_dataset, "validation")

    # Set prefetch for the dataset
    print("\n---- Setting prefetch for the dataset...")
    train_dataset = set_prefetch(train_dataset)
    val_dataset = set_prefetch(val_dataset)

    # Normalize data
    print("\n---- Normalizing dataset...")
    train_dataset_normalized = normalize_dataset(train_dataset)
    val_dataset_normalized = normalize_dataset(val_dataset)

    show_first_data_in_dataset(train_dataset_normalized, train_classes)

    # Check if a fit model is already saved
    if os.path.exists(results_dir + "cnn_first_model.keras"):
        print("\n---- A model is already saved !")
        print("\n---- Loading the model...")
        model = tf.keras.models.load_model(results_dir + "cnn_first_model.keras")
        model.summary()
    else:
        # Build the model
        print("\n---- Building the model...")
        model = create_model(train_classes, IMAGE_SIZE, IMAGE_SIZE)
        model = compile_mode(model)

        # Fit the model
        print("\n---- Training the model...")
        hist = train_model(model, train_dataset_normalized, val_dataset_normalized, EPOCHS)
        show_model_fit(hist, EPOCHS)

        # Save the model
        print("\n---- Saving the model...")
        save_model(model, "cnn_first_model", "keras")

    # Evaluate the model
    print("\n---- Evaluating the model...")
    print("### Val dataset evaluation:")
    model.evaluate(val_dataset_normalized)

    print("### Train dataset evaluation:")
    model.evaluate(train_dataset_normalized)

    # Quantize the model
    print("\n---- Quantizing the model...")

    # TODO : refactor print to logger
    # TODO : add test dataset
    # TODO : add inference function










