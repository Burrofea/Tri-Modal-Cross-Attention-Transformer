"""Unit tests for inference output schema and validation."""
import unittest
import torch
from src.inference.predictor import DeepfakePredictor

class TestInference(unittest.TestCase):
    def test_predictor_initialization(self):
        predictor = DeepfakePredictor(device="cpu")
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.audio_prep)
        self.assertIsNotNone(predictor.rppg_prep)
        self.assertIsNotNone(predictor.video_prep)

    def test_nonexistent_video_handling(self):
        predictor = DeepfakePredictor(device="cpu")
        res = predictor.predict_video("non_existent_video_file_xyz123.mp4")
        self.assertEqual(res["status"], "failed")
        self.assertIn("error", res)

if __name__ == "__main__":
    unittest.main()
