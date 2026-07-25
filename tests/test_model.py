"""
Unit tests for model architecture factory (Custom CNN & MobileNetV2).
"""

import unittest
import numpy as np
from src.model import build_custom_cnn, build_mobilenet_v2, get_model


class TestModelArchitectures(unittest.TestCase):
    def test_custom_cnn_output_shape(self):
        model = build_custom_cnn(input_shape=(150, 150, 3), num_classes=2)
        dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2))
        self.assertAlmostEqual(np.sum(output.numpy()), 1.0, places=5)

    def test_mobilenetv2_output_shape(self):
        model = build_mobilenet_v2(input_shape=(150, 150, 3), num_classes=2)
        dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2))
        self.assertAlmostEqual(np.sum(output.numpy()), 1.0, places=5)

    def test_get_model_factory(self):
        cnn_model = get_model("custom_cnn")
        self.assertEqual(cnn_model.name, "Custom_CNN_FaceMaskDetector")
        
        mobilenet_model = get_model("mobilenetv2")
        self.assertEqual(mobilenet_model.name, "MobileNetV2_FaceMaskDetector")

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            get_model("invalid_model_name")


if __name__ == "__main__":
    unittest.main()
