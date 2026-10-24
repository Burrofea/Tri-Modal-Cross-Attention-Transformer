"""Audio extraction and Log-Mel spectrogram computation."""
import subprocess
from typing import Optional, Tuple
import numpy as np
import scipy.signal as sps

def create_mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, f_min: float = 0.0, f_max: Optional[float] = None) -> np.ndarray:
    """
    Constructs a triangular Mel-filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    f_max = f_max or (sr / 2.0)
    # Convert Hz to Mel
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for f in range(left, center):
            if center > left and f < n_freqs:
                fbank[i - 1, f] = (f - left) / (center - left)
        for f in range(center, right):
            if right > center and f < n_freqs:
                fbank[i - 1, f] = (right - f) / (right - center)

    return fbank


class AudioPreprocessor:
    """
    Extracts raw audio tracks from video via FFmpeg and converts them to log-Mel spectrograms.
    """
    def __init__(self, sample_rate: int = 16000, duration_sec: float = 5.0, n_mels: int = 64, n_fft: int = 400, hop_length: int = 160, max_frames: int = 500):
        self.sr = sample_rate
        self.duration_sec = duration_sec
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.fbank = create_mel_filterbank(self.sr, self.n_fft, self.n_mels)

    def extract_audio_pcm(self, video_path: str) -> Tuple[np.ndarray, bool]:
        """
        Runs ffmpeg to extract 16kHz mono float32 PCM audio.
        Returns: (audio_waveform, has_audio)
        """
        cmd = [
            "ffmpeg", "-v", "quiet", "-i", video_path,
            "-t", str(self.duration_sec),
            "-vn", "-ac", "1", "-ar", str(self.sr),
            "-f", "f32le", "-"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, check=True)
            audio = np.frombuffer(res.stdout, dtype=np.float32).copy()
            if len(audio) == 0:
                # No audio track present
                target_len = int(self.duration_sec * self.sr)
                return np.zeros(target_len, dtype=np.float32), False
            # Pad or truncate to target duration
            target_len = int(self.duration_sec * self.sr)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            return audio, True
        except Exception:
            target_len = int(self.duration_sec * self.sr)
            return np.zeros(target_len, dtype=np.float32), False

    def compute_log_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes log-Mel spectrogram (n_mels, max_frames) from 1D PCM audio array.
        """
        _, _, zxx = sps.stft(
            audio,
            fs=self.sr,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            boundary=None,
        )
        power_spec = (np.abs(zxx) ** 2).astype(np.float32)  # (n_fft // 2 + 1, time_steps)
        mel_spec = np.dot(self.fbank, power_spec)  # (n_mels, time_steps)
        log_mel = np.log(mel_spec + 1e-6)

        # Ensure exact frame width (max_frames)
        if log_mel.shape[1] < self.max_frames:
            pad_width = self.max_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=np.log(1e-6))
        else:
            log_mel = log_mel[:, :self.max_frames]

        return log_mel.astype(np.float32)

    def process(self, video_path: str) -> Tuple[np.ndarray, bool]:
        audio, has_audio = self.extract_audio_pcm(video_path)
        mel = self.compute_log_mel(audio)
        return mel, has_audio
