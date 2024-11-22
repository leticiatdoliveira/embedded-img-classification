# src/utils/preprocessing.py
import cv2
import numpy as np
from torchvision import transforms
import torch
from .config import mean_channels, std_channels


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize the input image to the desired size.

    :param image: Input image in numpy array format
    :param size: Desired size to resize the image
    :return: Resized image
    """
    return cv2.resize(image, (size, size))


def normalize_image(image: np.ndarray) -> torch.Tensor:
    """
    Normalize the input image using torchvision transforms.

    :param image: Input image in numpy array format
    :return: Normalized image as a torch tensor
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_channels,
                             std=std_channels)
    ])
    return preprocess(image)


def preprocess_image(image: np.ndarray, size: int) -> torch.Tensor:
    """
    Preprocess the input image by resizing and normalizing.

    :param image: Input image in numpy array format
    :param size: Desired size to resize the image
    :return: Preprocessed image as a torch tensor
    """
    resized_image = resize_image(image, size)
    normalized_image = normalize_image(resized_image)
    return normalized_image
