# tests/test_mobilenetv2.py
import unittest
import numpy as np
import torch
from src.models.mobilenetv2_model import MobileNetV2Model


class TestMobileNetV2Model(unittest.TestCase):
    def setUp(self):
        self.model = MobileNetV2Model()

    def test_preprocess(self):
        image = np.random.rand(224, 224, 3).astype(np.float32)
        tensor = self.model.preprocess(image)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_predict(self):
        image = np.random.rand(224, 224, 3).astype(np.float32)
        tensor = self.model.preprocess(image)
        output = self.model.predict(tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[1], 1000)  # Assuming 1000 classes

    def test_get_top_predictions(self):
        output = torch.rand(1, 1000)
        top_predictions = get_top_predictions(output, top_k=2)
        self.assertIsInstance(top_predictions, list)
        self.assertEqual(len(top_predictions), 2)


if __name__ == '__main__':
    unittest.main()