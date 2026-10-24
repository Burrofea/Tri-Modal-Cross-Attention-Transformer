"""Unit tests for audio, video, and rPPG preprocessing pipelines."""
import unittest
import numpy as np
import scipy.signal as sps
import torch

from src.preprocessing.audio import AudioPreprocessor, create_mel_filterbank
from src.preprocessing.rppg import RPPGPreprocessor
from src.preprocessing.video import VideoPreprocessor

class TestPreprocessing(unittest.TestCase):
    def test_mel_filterbank_shape(self):
        sr, n_fft, n_mels = 16000, 400, 64
        fbank = create_mel_filterbank(sr, n_fft, n_mels)
        self.assertEqual(fbank.shape, (64, 201))
        # Ensure filterbank weights are non-negative and normalized
        self.assertTrue(np.all(fbank >= 0.0))

    def test_audio_log_mel_computation(self):
        prep = AudioPreprocessor(sample_rate=16000, duration_sec=5.0, n_mels=64)
        synthetic_pcm = np.random.randn(16000 * 5).astype(np.float32)
        log_mel = prep.compute_log_mel(synthetic_pcm)
        self.assertEqual(log_mel.shape, (64, 500))
        self.assertFalse(np.isnan(log_mel).any())

    def test_pos_algorithm_pulse_extraction(self):
        fps = 30.0
        T = int(5.0 * fps)
        time_axis = np.arange(T) / fps

        # Simulate subtle cardiac pulse at 1.2 Hz (72 BPM) superimposed on skin reflectance
        cardiac_signal = 0.02 * np.sin(2 * np.pi * 1.2 * time_axis)
        r_trace = 180.0 + 5.0 * cardiac_signal + np.random.randn(T) * 0.1
        g_trace = 120.0 + 10.0 * cardiac_signal + np.random.randn(T) * 0.1
        b_trace = 95.0 + 3.0 * cardiac_signal + np.random.randn(T) * 0.1
        rgb_trace = np.stack([r_trace, g_trace, b_trace], axis=1).astype(np.float32)

        pulse = RPPGPreprocessor.pos_algorithm(rgb_trace, fps)
        self.assertEqual(len(pulse), T)
        self.assertFalse(np.isnan(pulse).any())

    def test_butterworth_bandpass_filtering(self):
        fps = 30.0
        T = int(5.0 * fps)
        time_axis = np.arange(T) / fps
        nyq = fps / 2.0
        low = 0.5 / nyq
        high = 4.0 / nyq
        b, a = sps.butter(3, [low, high], btype="band")

        # Mixed signal: 0.1 Hz drift (respiration/motion) + 1.5 Hz pulse + 10 Hz high-frequency noise
        mixed = np.sin(2 * np.pi * 0.1 * time_axis) + np.sin(2 * np.pi * 1.5 * time_axis) + np.sin(2 * np.pi * 10.0 * time_axis)
        filtered = sps.filtfilt(b, a, mixed)

        self.assertEqual(len(filtered), T)
        # Power should be concentrated in passband
        fft_orig = np.abs(np.fft.rfft(mixed))
        fft_filt = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(T, d=1.0 / fps)

        # 0.1 Hz component and 10 Hz component should be heavily attenuated
        idx_01 = np.argmin(np.abs(freqs - 0.1))
        idx_15 = np.argmin(np.abs(freqs - 1.5))
        idx_10 = np.argmin(np.abs(freqs - 10.0))

        self.assertGreater(fft_filt[idx_15] / (fft_filt[idx_01] + 1e-6), 5.0)
        self.assertGreater(fft_filt[idx_15] / (fft_filt[idx_10] + 1e-6), 5.0)

    def test_video_preprocessor_embeddings(self):
        video_prep = VideoPreprocessor(embed_dim=512, backbone="resnet18", device="cpu")
        synthetic_crops = np.random.randint(0, 255, size=(16, 128, 128, 3), dtype=np.uint8)
        embeddings = video_prep.embed_crops(synthetic_crops)
        self.assertEqual(embeddings.shape, (16, 512))
        self.assertFalse(np.isnan(embeddings).any())

if __name__ == "__main__":
    unittest.main()
