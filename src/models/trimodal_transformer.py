"""Full Tri-Modal Cross-Attention Transformer with Dynamic Gating and Multi-Task Heads."""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import AudioTokenizer, VideoTokenizer, RPPGTokenizer
from .cross_attention import CrossAttentionBlock
from .gating import DynamicGatingModule

NEG_INF = -1e4

class TriModalTransformer(nn.Module):
    """
    Genuine Tri-Modal Cross-Attention Transformer.
    Integrates Video, Audio, and Physiological rPPG streams with reliability-aware dynamic gating.
    Produces multi-task predictions:
      - video_fake (B,)
      - audio_fake (B,)
      - any_fake   (B,)
    """
    def __init__(
        self,
        hidden_dim: int = 192,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        audio_tokens: int = 25,
        video_tokens: int = 16,
        rppg_windows: int = 5,
        rppg_feat_dim: int = 147,
        gate_input_dim: int = 4,
        use_dynamic_gating: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_tokens = audio_tokens
        self.video_tokens = video_tokens
        self.rppg_windows = rppg_windows
        self.use_dynamic_gating = use_dynamic_gating

        # Tokenizers
        self.audio_tok = AudioTokenizer(hidden_dim, audio_tokens)
        self.video_tok = VideoTokenizer(512, hidden_dim, video_tokens)
        self.rppg_tok = RPPGTokenizer(rppg_feat_dim, hidden_dim, rppg_windows)

        # Dynamic Gating
        self.gate_module = DynamicGatingModule(gate_input_dim, hidden_dim=32)

        # Cross-Attention Layers (3 blocks per layer: Audio, Video, rPPG)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttentionBlock(hidden_dim, num_heads, dropout),  # Audio attends to [Video, rPPG]
                CrossAttentionBlock(hidden_dim, num_heads, dropout),  # Video attends to [Audio, rPPG]
                CrossAttentionBlock(hidden_dim, num_heads, dropout),  # rPPG attends to [Audio, Video]
            ])
            for _ in range(num_layers)
        ])

        # Normalization
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_r = nn.LayerNorm(hidden_dim)

        # Unimodal Auxiliary Heads (for Stage 1 warmup and baseline comparison)
        self.aux_a = nn.Linear(hidden_dim, 1)  # Predicts audio_fake
        self.aux_v = nn.Linear(hidden_dim, 1)  # Predicts video_fake
        self.aux_r = nn.Linear(hidden_dim, 1)  # Predicts video_fake (from physiological pulse)

        # Multi-task Classifier Head
        # Inputs: pooled_audio (D) + pooled_video (D) + pooled_rppg (D) + availability_flags (3)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # Logits for: [video_fake, audio_fake, any_fake]
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        stage: int = 2,
        disable_modalities: Tuple[str, ...] = (),
        use_dynamic_gating: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary containing 'mel', 'emb', 'tok', 'rm', 'gate_in'
            stage: 1 (unimodal auxiliary training) or 2 (full tri-modal cross-attention)
            disable_modalities: Tuple of modalities to zero out (e.g. ('rppg',), ('audio',))
            use_dynamic_gating: Override default dynamic gating behavior
        Returns:
            Dictionary with logits: 'logits_video', 'logits_audio', 'logits_any', 'aux_a', 'aux_v', 'aux_r', 'gate'
        """
        mel = batch["mel"]
        emb = batch["emb"]
        tok = batch["tok"]
        rm = batch["rm"].bool()
        gate_in = batch["gate_in"]

        b_sz = mel.shape[0]
        device = mel.device

        # Tokenization
        A = self.audio_tok(mel)  # (B, 25, D)
        V = self.video_tok(emb)  # (B, 16, D)
        R = self.rppg_tok(tok)   # (B, 5, D)

        # Dynamic Gating
        apply_gate = self.use_dynamic_gating if (use_dynamic_gating is None) else use_dynamic_gating
        if apply_gate:
            g = self.gate_module(gate_in, rm)  # (B, 5, 1)
        else:
            g = rm.unsqueeze(-1).float()  # Binary mask only

        # Attenuate rPPG tokens by reliability weight
        R = R * g
        g_sum = g.sum(dim=1).clamp_min(1e-6)
        pr_unimodal = R.sum(dim=1) / g_sum

        out: Dict[str, torch.Tensor] = {
            "gate": g.squeeze(-1),
            "aux_a": self.aux_a(A.mean(dim=1)).squeeze(-1),
            "aux_v": self.aux_v(V.mean(dim=1)).squeeze(-1),
            "aux_r": self.aux_r(pr_unimodal).squeeze(-1),
        }

        if stage == 1:
            return out

        # Modality availability & dropout during training
        keep_a = torch.ones(b_sz, dtype=torch.bool, device=device)
        keep_v = torch.ones(b_sz, dtype=torch.bool, device=device)
        keep_r = torch.ones(b_sz, dtype=torch.bool, device=device)

        if self.training:
            # Modality dropout to promote robustness under missing streams
            keep_a = torch.rand(b_sz, device=device) > 0.15
            keep_v = (torch.rand(b_sz, device=device) > 0.10) | (~keep_a)
            keep_r = torch.rand(b_sz, device=device) > 0.20

        # Explicit test-time disabling for ablations
        if "audio" in disable_modalities:
            keep_a = keep_a & False
        if "video" in disable_modalities:
            keep_v = keep_v & False
        if "rppg" in disable_modalities:
            keep_r = keep_r & False

        # rPPG is also marked inactive if all windows are masked/invalid
        keep_r = keep_r & (g.squeeze(-1).sum(dim=1) > 0)

        # Construct attention masks/biases
        # Valid tokens get 0.0, masked tokens get NEG_INF
        mask_fn = lambda keep, length: torch.zeros(b_sz, length, device=device).masked_fill(~keep.unsqueeze(1), NEG_INF)
        bias_a = mask_fn(keep_a, self.audio_tokens)
        bias_v = mask_fn(keep_v, self.video_tokens)

        # Soft gating attention bias for rPPG
        # When g is close to 0, log(g) becomes large negative, softly dampening attention
        soft_bias_r = torch.log(g.squeeze(-1).clamp_min(1e-4)).masked_fill(~(rm & keep_r.unsqueeze(1)), NEG_INF)

        # Helper to format attention bias for SDPA: (B, 1, 1, L_context)
        bias_cat = lambda *tensors: torch.cat(tensors, dim=1)[:, None, None, :]

        # Tri-Modal Cross-Attention Blocks
        for x_a, x_v, x_r in self.layers:
            # Audio queries [Video, rPPG]
            ctx_for_a = torch.cat([V, R], dim=1)
            bias_for_a = bias_cat(bias_v, soft_bias_r)
            A = x_a(A, ctx_for_a, bias_for_a)

            # Video queries [Audio, rPPG]
            ctx_for_v = torch.cat([A, R], dim=1)
            bias_for_v = bias_cat(bias_a, soft_bias_r)
            V = x_v(V, ctx_for_v, bias_for_v)

            # rPPG queries [Audio, Video]
            ctx_for_r = torch.cat([A, V], dim=1)
            bias_for_r = bias_cat(bias_a, bias_v)
            R = x_r(R, ctx_for_r, bias_for_r)

        A = self.norm_a(A)
        V = self.norm_v(V)
        R = self.norm_r(R)

        # Pooling
        pool_a = A.mean(dim=1) * keep_a[:, None].float()
        pool_v = V.mean(dim=1) * keep_v[:, None].float()

        valid_r_weights = (soft_bias_r > (NEG_INF / 2.0)).float().unsqueeze(-1)
        pool_r = (R * valid_r_weights).sum(dim=1) / (valid_r_weights.sum(dim=1) + 1e-6)
        pool_r = pool_r * keep_r[:, None].float()

        flags = torch.stack([keep_a, keep_v, keep_r], dim=1).float()
        fused_repr = torch.cat([pool_a, pool_v, pool_r, flags], dim=1)

        logits = self.head(fused_repr)  # (B, 3)

        out["logits_video"] = logits[:, 0]
        out["logits_audio"] = logits[:, 1]
        out["logits_any"] = logits[:, 2]

        return out
