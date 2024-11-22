# src/models/mobilenetv2_model.py
import torch
from torch.jit import RecursiveScriptModule
from torchvision import models
from .base_model import BaseModel


class MobileNetV2Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
        self.model = torch.jit.script(self.model)

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(tensor.unsqueeze(0))

    def get_top_predictions(self, output: torch.Tensor, top_k: int) -> list:
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_idxs = torch.topk(probabilities, top_k)
        return [(idx.item(), prob.item()) for idx, prob in zip(top_idxs, top_probs)]