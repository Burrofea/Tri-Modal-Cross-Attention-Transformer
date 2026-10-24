"""Cross-attention Transformer blocks for inter-modal feature exchange."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention between query modality and context modalities.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, context: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: (B, L_q, D)
            context: (B, L_k, D)
            attn_bias: Optional (B, 1, 1, L_k) or (B, 1, L_q, L_k) additive mask
        Returns:
            out: (B, L_q, D)
        """
        b_sz, l_q, _ = query.shape
        _, l_k, _ = context.shape

        q = self.q_proj(query).view(b_sz, l_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b_sz, l_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b_sz, l_k, self.num_heads, self.head_dim).transpose(1, 2)

        drop_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop_p)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b_sz, l_q, self.embed_dim)
        return self.out_proj(attn_out)


class CrossAttentionBlock(nn.Module):
    """
    A complete cross-attention block with LayerNorm, Pre-LN residuals, and a 2-layer MLP.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ctx = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.drop_attn = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        # Cross-attention residual
        q_norm = self.norm_q(query)
        ctx_norm = self.norm_ctx(context)
        attended = self.attn(q_norm, ctx_norm, attn_bias)
        x = query + self.drop_attn(attended)

        # FFN residual
        x = x + self.mlp(self.norm_mlp(x))
        return x
