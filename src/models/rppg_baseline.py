"""Standalone rPPG-only deepfake detection baseline model."""
from typing import Dict
import torch
import torch.nn as nn
from .tokenizers import RPPGTokenizer

class RPPGBaselineModel(nn.Module):
    """
    Physiological rPPG unimodal baseline.
    Classifies video authenticity using strictly facial blood-volume-pulse (BVP) frequency tokens
    and signal validity masks, without access to audio or visual artifact streams.
    """
    def __init__(self, in_dim: int = 147, hidden_dim: int = 128, num_windows: int = 5, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = RPPGTokenizer(in_dim, hidden_dim, num_windows)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tok = batch["tok"]
        rm = batch["rm"].bool()

        # Tokenize
        x = self.tokenizer(tok)  # (B, NW, hidden_dim)

        # Transformer encoder with key_padding_mask (~rm)
        padding_mask = ~rm  # True indicates elements that should be masked
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)

        # Pool over valid windows
        mask_weights = rm.float().unsqueeze(-1)
        denom = mask_weights.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * mask_weights).sum(dim=1) / denom

        logits_video = self.classifier(pooled).squeeze(-1)

        return {
            "logits_video": logits_video,
            "logits_any": logits_video,  # In rPPG only, any fake = video fake
            "probabilities": torch.sigmoid(logits_video),
        }
