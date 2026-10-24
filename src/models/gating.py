"""Dynamic reliability gating module for physiological signals and quality weighting."""
import torch
import torch.nn as nn

class DynamicGatingModule(nn.Module):
    """
    Computes per-sample, per-window reliability weights from signal quality metadata
    (SNR, video bitrate, image sharpness, face detection coverage).
    Ensures invalid windows (Case A: missing face, corrupt crops) are strictly masked to zero weight.
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, gate_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_in: (B, NW, 4) tensor of [SNR/10, log_bitrate, log_sharpness, face_detect_rate]
            mask: (B, NW) boolean tensor indicating valid rPPG extraction
        Returns:
            g: (B, NW, 1) reliability weights bounded in [0, 1]
        """
        raw_gate = self.net(gate_in)  # (B, NW, 1)
        # Strictly zero out masked / invalid windows (e.g. Case A: no face found)
        valid_mask = mask.unsqueeze(-1).float()
        effective_gate = raw_gate * valid_mask
        return effective_gate
