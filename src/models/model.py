# src/models/model.py
import torch
from torchvision import models


class Model:
    def __init__(self, model_type: str, apply_quantize: bool, apply_jit: bool):
        self.model_name = model_type
        self.quantized = apply_quantize
        self.jit = apply_jit

        if self.quantized:
            torch.backends.quantized.engine = 'qnnpack'

    def load_pretrained_model(self):
        """
        Load the pre-trained model based on the model type

        :return: Pre-trained model
        """
        if self.model_name == "modelnet_v2":
            pre_trained_model = models.quantization.mobilenet_v2(pretrained=True, quantize=self.quantized)
            torch.jit.script(pre_trained_model) if self.jit else pre_trained_model
        else:
            pre_trained_model = None

        return pre_trained_model

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the input image

        :param image: Input image tensor
        :return: Model output tensor
        """
        return self.load_pretrained_model()(image)

    @staticmethod
    def get_top_predictions(model_output: torch.Tensor, top_k: int) -> list:
        """
        Get the top k class predictions

        :param model_output: Model output tensor
        :param top_k: Number of top classes to return
        :return: Tuple collections of top classes indices and their softmax probabilities
        """
        # get softmax probabilities
        probabilities = torch.nn.functional.softmax(model_output[0], dim=0)
        # get the largest k classes and their probabilities
        top_probs, top_idxs = torch.topk(probabilities, top_k)
        return [(idx.item(), prob.item()) for idx, prob in zip(top_idxs, top_probs)]
