"""Default configuration parameters for Multimodal Deepfake Detector."""
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration_sec: float = 5.0
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    max_frames: int = 500

@dataclass
class VideoConfig:
    num_frames: int = 16
    crop_size: int = 128
    embed_dim: int = 512
    backbone: str = "resnet18"  # Options: "resnet18", "mobilenet_v3"

@dataclass
class RPPGConfig:
    bandpass_hz: Tuple[float, float] = (0.5, 4.0)
    filter_order: int = 3
    window_sec: float = 3.0
    hop_sec: float = 0.5
    num_windows: int = 5
    num_freq_bins: int = 48
    landmarks: Tuple[int, int, int] = (205, 425, 151)  # Left cheek, right cheek, forehead
    min_face_detect_rate: float = 0.65
    roi_scale: float = 0.06

@dataclass
class ModelConfig:
    hidden_dim: int = 192
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    audio_tokens: int = 25
    video_tokens: int = 16
    rppg_tokens: int = 5
    rppg_feat_dim: int = 147  # 3 ROIs * 48 bins + 3 correlation values
    gate_input_dim: int = 4   # [SNR/10, log_bitrate, log_sharpness, face_detect_rate]
    use_dynamic_gating: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 4
    seed: int = 42
    device: str = "cuda"
    modality_dropout_prob: float = 0.15
    consistency_loss_weight: float = 0.1

@dataclass
class ProjectConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    rppg: RPPGConfig = field(default_factory=RPPGConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
