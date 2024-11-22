# src/models/base_model.py
import torch
import numpy as np
from src.utils.preprocessing import preprocess_image


class BaseModel:

    def preprocess(self, image: np.ndarray, size: int) -> torch.Tensor:
        return preprocess_image(image, size)

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")

    def get_top_predictions(self, output: torch.Tensor, top_k: int) -> list:
        raise NotImplementedError("Subclasses should implement this method")