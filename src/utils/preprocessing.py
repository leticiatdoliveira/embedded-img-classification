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


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert the input image from BGR to RGB format.

    :param image: Input image in numpy array format
    :return: Image in RGB format
    """
    return image[:, :, [2, 1, 0]]
