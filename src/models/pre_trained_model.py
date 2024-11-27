# src/models/pre_trained_model.py
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
        elif self.model_name == "resnet18":
            pre_trained_model = models.quantization.resnet18(pretrained=True, quantize=self.quantized)
        elif self.model_name == "yolov5":
            # as there aren't official a hub pre-trained model, we will use the ultralytics/yolov5 repository
            pre_trained_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # force quantization
            if self.quantized:
                pre_trained_model.fuse()  # Fuse Conv, BN, and ReLU layers
                pre_trained_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
                torch.quantization.prepare(pre_trained_model, inplace=True)
                torch.quantization.convert(pre_trained_model, inplace=True)
        else:
            pre_trained_model = None

        torch.jit.script(pre_trained_model) if self.jit else pre_trained_model
        return pre_trained_model

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the input image

        :param image: Input image tensor
        :return: Model output tensor
        """
        model = self.load_pretrained_model()
        if self.model_name == "yolov5":
            results = model(image)
            output = results.xyxy[0]
        else:
            output = model(image)
        return output

    def get_top_predictions(self, model_output: torch.Tensor, top_k: int) -> list:
        """
        Get the top k class predictions

        :param model_output: Model output tensor
        :param top_k: Number of top classes to return
        :return: Tuple collections of top classes indices and their softmax probabilities
        """
        if self.model_name == "yolov5":
            top_probs, top_idxs = torch.topk(model_output[:, 4], top_k)  # confidence scores
            return [(model_output[idx].tolist(), prob.item()) for idx, prob in zip(top_idxs, top_probs)]
        else:
            probabilities = torch.nn.functional.softmax(model_output[0], dim=0)
            top_probs, top_idxs = torch.topk(probabilities, top_k)
            return [(idx.item(), prob.item()) for idx, prob in zip(top_idxs, top_probs)]

