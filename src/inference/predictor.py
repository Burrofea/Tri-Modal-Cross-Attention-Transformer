"""High-level single-video and batch inference pipeline."""
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from ..preprocessing.audio import AudioPreprocessor
from ..preprocessing.rppg import RPPGPreprocessor
from ..preprocessing.video import VideoPreprocessor
from ..models.trimodal_transformer import TriModalTransformer

class DeepfakePredictor:
    """
    End-to-end inference engine for evaluating arbitrary input videos.
    Extracts multimodal streams in real time and produces calibrated authenticity predictions.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: str = "cuda",
        threshold_any: float = 0.5,
        threshold_video: float = 0.5,
        threshold_audio: float = 0.5,
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.threshold_any = threshold_any
        self.threshold_video = threshold_video
        self.threshold_audio = threshold_audio

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.model = TriModalTransformer().to(self.device)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state_dict)
        else:
            # Fallback to freshly instantiated architecture
            self.model = TriModalTransformer().to(self.device)

        self.model.eval()

        # Preprocessors
        self.audio_prep = AudioPreprocessor()
        self.rppg_prep = RPPGPreprocessor()
        self.video_prep = VideoPreprocessor(device="cpu")

    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict:
        """
        Executes end-to-end inference on a single video file.
        Returns detailed structured dictionary with authenticity probabilities, quality metrics, and warnings.
        """
        if not os.path.exists(video_path):
            return {
                "error": f"File not found: {video_path}",
                "status": "failed"
            }

        warnings: List[str] = []

        # 1. Audio stream
        mel, has_audio = self.audio_prep.process(video_path)
        if not has_audio:
            warnings.append("Video does not contain an audible audio track; audio authenticity inference unavailable.")

        # 2. rPPG and Face extraction
        rppg_out = self.rppg_prep.process(video_path)
        crops = rppg_out["crops"]
        rppg_tok = rppg_out["rppg_tok"]
        rppg_mask = rppg_out["rppg_mask"]
        snr = rppg_out["snr"]
        quality = rppg_out["quality"]

        face_detected = bool(np.any(rppg_mask))
        if not face_detected:
            warnings.append("No stable human face detected (Case A). Physiological rPPG weight attenuated to 0.")

        # 3. Video visual embedding
        emb = self.video_prep.embed_crops(crops)

        # 4. Format gating input
        norm_q = (quality - np.array([6.0, 3.0, 0.5], dtype=np.float32)) / np.array([2.0, 2.0, 0.5], dtype=np.float32)
        snr_feat = np.clip(snr, -20.0, 20.0)[:, None] / 10.0
        q_tile = np.tile(norm_q[None, :], (len(snr), 1))
        gate_in = np.concatenate([snr_feat, q_tile], axis=1).astype(np.float32)

        # Normalize audio mel
        mel_norm = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

        # Construct batch of size 1
        batch = {
            "mel": torch.from_numpy(mel_norm).unsqueeze(0).to(self.device),
            "emb": torch.from_numpy(emb).unsqueeze(0).to(self.device),
            "tok": torch.from_numpy(rppg_tok).unsqueeze(0).to(self.device),
            "rm": torch.from_numpy(rppg_mask).unsqueeze(0).to(self.device),
            "gate_in": torch.from_numpy(gate_in).unsqueeze(0).to(self.device),
        }

        # Inference forward pass
        disable = ()
        if not has_audio:
            disable = ("audio",)

        outputs = self.model(batch, stage=2, disable_modalities=disable)

        p_v = float(torch.sigmoid(outputs["logits_video"])[0].cpu().item())
        p_a = float(torch.sigmoid(outputs["logits_audio"])[0].cpu().item()) if has_audio else 0.0
        p_any = float(torch.sigmoid(outputs["logits_any"])[0].cpu().item())

        is_video_fake = bool(p_v >= self.threshold_video)
        is_audio_fake = bool(p_a >= self.threshold_audio) if has_audio else False
        is_any_fake = bool(p_any >= self.threshold_any)

        avg_snr = float(np.mean(snr[rppg_mask])) if np.any(rppg_mask) else 0.0

        return {
            "status": "success",
            "video_path": video_path,
            "probabilities": {
                "video_fake_probability": round(p_v, 4),
                "audio_fake_probability": round(p_a, 4) if has_audio else None,
                "any_fake_probability": round(p_any, 4),
            },
            "predictions": {
                "video_fake": is_video_fake,
                "audio_fake": is_audio_fake if has_audio else None,
                "any_fake": is_any_fake,
                "verdict": "DEEPFAKE" if is_any_fake else "REAL",
            },
            "modality_availability": {
                "video_available": True,
                "audio_available": has_audio,
                "face_detected": face_detected,
                "rppg_valid_windows": int(np.sum(rppg_mask)),
            },
            "physiological_quality": {
                "mean_pulse_snr_db": round(avg_snr, 2),
                "face_coverage_rate": round(float(quality[2]), 3),
                "laplacian_sharpness": round(float(np.exp(quality[1]) - 1.0), 2),
            },
            "warnings": warnings,
            "system_info": {
                "device": str(self.device),
                "model": "TriModalCrossAttentionTransformer_v1",
            }
        }
