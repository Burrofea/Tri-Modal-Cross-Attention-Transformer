"""Multi-task loss with logical consistency regularization."""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskConsistencyLoss(nn.Module):
    """
    Computes multi-task BCE loss for (video_fake, audio_fake, any_fake) and enforces logical
    consistency: any_fake must correspond to (video_fake OR audio_fake).
    """
    def __init__(self, consistency_weight: float = 0.1, aux_weight: float = 0.2):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.aux_weight = aux_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, stage: int = 2) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dictionary containing logits and auxiliary predictions
            targets: (B, 3) tensor of [y_video, y_audio, y_any]
            stage: 1 (unimodal warmup) or 2 (tri-modal cross-attention)
        """
        y_video = targets[:, 0]
        y_audio = targets[:, 1]
        y_any = targets[:, 2]

        if stage == 1:
            loss_a = self.bce(outputs["aux_a"], y_audio)
            loss_v = self.bce(outputs["aux_v"], y_video)
            loss_r = self.bce(outputs["aux_r"], y_video)
            total_loss = loss_a + loss_v + loss_r
            return {
                "loss": total_loss,
                "loss_aux_a": loss_a,
                "loss_aux_v": loss_v,
                "loss_aux_r": loss_r,
            }

        # Stage 2: Main multi-task heads
        loss_v = self.bce(outputs["logits_video"], y_video)
        loss_a = self.bce(outputs["logits_audio"], y_audio)
        loss_any = self.bce(outputs["logits_any"], y_any)

        # Probabilities for logical consistency
        p_v = torch.sigmoid(outputs["logits_video"])
        p_a = torch.sigmoid(outputs["logits_audio"])
        p_any = torch.sigmoid(outputs["logits_any"])

        # Logical OR expectation: 1 - (1 - P(v)) * (1 - P(a))
        p_or_expected = 1.0 - (1.0 - p_v) * (1.0 - p_a)
        consistency_loss = F.mse_loss(p_any, p_or_expected)

        # Auxiliary regularization during stage 2
        aux_loss = (
            self.bce(outputs["aux_a"], y_audio) +
            self.bce(outputs["aux_v"], y_video) +
            self.bce(outputs["aux_r"], y_video)
        )

        total_loss = loss_v + loss_a + loss_any + (self.consistency_weight * consistency_loss) + (self.aux_weight * aux_loss)

        return {
            "loss": total_loss,
            "loss_video": loss_v,
            "loss_audio": loss_a,
            "loss_any": loss_any,
            "loss_consistency": consistency_loss,
            "loss_aux": aux_loss,
        }
