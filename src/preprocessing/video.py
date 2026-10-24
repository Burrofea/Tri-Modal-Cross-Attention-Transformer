"""Visual feature extractor for face crops."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VideoPreprocessor(nn.Module):
    """
    Extracts deep visual features from sampled face crops using a PyTorch visual backbone (ResNet-18).
    Produces (num_frames, 512) feature embeddings per video.
    """
    def __init__(self, embed_dim: int = 512, backbone: str = "resnet18", device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim

        if backbone == "resnet18":
            base_model = models.resnet18(weights=None)
            base_model.fc = nn.Identity()
            self.backbone = base_model
            in_dim = 512
        elif backbone == "mobilenet_v3":
            base_model = models.mobilenet_v3_small(weights=None)
            base_model.classifier = nn.Identity()
            self.backbone = base_model
            in_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if in_dim != embed_dim:
            self.proj = nn.Linear(in_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def embed_crops(self, crops_np: np.ndarray) -> np.ndarray:
        """
        Embeds (N, 128, 128, 3) uint8 image array to (N, 512) float32 embeddings.
        """
        t = torch.from_numpy(crops_np).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, 128, 128)
        t = F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)
        t = t.to(self.device)
        norm_t = (t - self.mean) / self.std

        features = self.backbone(norm_t)
        embeddings = self.proj(features)
        return embeddings.cpu().numpy().astype(np.float32)
