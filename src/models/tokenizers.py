"""Tokenizer layers projecting raw streams into shared latent space with positional encodings."""
import torch
import torch.nn as nn

class AudioTokenizer(nn.Module):
    """
    Transforms log-mel spectrograms (B, 64, 500) into sequence of 25 acoustic tokens in R^hidden_dim.
    """
    def __init__(self, hidden_dim: int = 192, num_tokens: int = 25):
        super().__init__()
        self.num_tokens = num_tokens

        def conv_block(in_c, out_c, pool):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        # Output shape after 3 blocks: (B, 128, 8, 25)
        self.cnn = nn.Sequential(
            conv_block(1, 32, (2, 2)),
            conv_block(32, 64, (2, 2)),
            conv_block(64, 128, (2, 5)),
        )
        self.proj = nn.Linear(128 * 8, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 64, 500) -> (B, 1, 64, 500)
        x = self.cnn(mel.unsqueeze(1))  # (B, 128, 8, 25)
        x = x.permute(0, 3, 1, 2).flatten(2)  # (B, 25, 1024)
        tokens = self.proj(x) + self.pos_embed
        return tokens


class VideoTokenizer(nn.Module):
    """
    Transforms deep visual embeddings (B, 16, 512) into 16 visual tokens in R^hidden_dim.
    """
    def __init__(self, in_dim: int = 512, hidden_dim: int = 192, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(emb) + self.pos_embed
        return tokens


class RPPGTokenizer(nn.Module):
    """
    Transforms rPPG frequency/correlation tokens (B, 5, 147) into 5 physiological tokens in R^hidden_dim.
    """
    def __init__(self, in_dim: int = 147, hidden_dim: int = 192, num_windows: int = 5):
        super().__init__()
        self.num_windows = num_windows
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_windows, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, rppg_feats: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(rppg_feats) + self.pos_embed
        return tokens
