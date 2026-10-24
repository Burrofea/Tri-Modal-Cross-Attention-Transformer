"""Unit tests for dataset scanning and speaker-disjoint splitting."""
import unittest
import pandas as pd
from src.data.scanner import AuthenticityLabels
from src.data.splitter import GroupedSplitter

class TestScannerAndSplits(unittest.TestCase):
    def test_category_labels(self):
        self.assertEqual(AuthenticityLabels.get_labels("RealVideo-RealAudio"), (0, 0, 0))
        self.assertEqual(AuthenticityLabels.get_labels("FakeVideo-RealAudio"), (1, 0, 1))
        self.assertEqual(AuthenticityLabels.get_labels("RealVideo-FakeAudio"), (0, 1, 1))
        self.assertEqual(AuthenticityLabels.get_labels("FakeVideo-FakeAudio"), (1, 1, 1))

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError):
            AuthenticityLabels.get_labels("Invalid-Category")

    def test_speaker_disjoint_splitting(self):
        # Create synthetic dataset with 20 distinct speakers and multiple clips per speaker
        rows = []
        for i in range(20):
            spk = f"id{i:04d}"
            for c in ["RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]:
                rows.append({
                    "uid": f"v{len(rows):04d}",
                    "speaker": spk,
                    "category": c,
                    "race": "Caucasian",
                    "gender": "men",
                    "video_fake": AuthenticityLabels.get_labels(c)[0],
                    "audio_fake": AuthenticityLabels.get_labels(c)[1],
                    "any_fake": AuthenticityLabels.get_labels(c)[2],
                })
        df = pd.DataFrame(rows)

        splitter = GroupedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        df_split = splitter.split(df, group_col="speaker")

        train_spk = set(df_split[df_split.split == "train"].speaker)
        val_spk = set(df_split[df_split.split == "val"].speaker)
        test_spk = set(df_split[df_split.split == "test"].speaker)

        # Assert mutual exclusivity across all splits (zero biometric leakage)
        self.assertEqual(len(train_spk & val_spk), 0, "Train and Val share speakers!")
        self.assertEqual(len(train_spk & test_spk), 0, "Train and Test share speakers!")
        self.assertEqual(len(val_spk & test_spk), 0, "Val and Test share speakers!")

if __name__ == "__main__":
    unittest.main()
