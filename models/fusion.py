# 全局、局部融合
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SemanticStructureFusion(nn.Module):
    """Original attention-based global-local fusion.

    Used by the image backbone to produce the global fused image feature. This
    branch is intentionally kept unchanged for quality prediction, because your
    experiments showed that replacing the whole fusion module with Mamba/V-Mamba
    hurts quality scores.
    """

    def __init__(self, semantic_dim: int = 768, local_dim: int = 256, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.semantic_proj = nn.Linear(semantic_dim, embed_dim)
        self.local_proj = nn.Conv2d(local_dim, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, f_sem: torch.Tensor, f_loc: torch.Tensor) -> torch.Tensor:
        # f_sem: [B, D], f_loc: [B, C, H, W]
        q = self.semantic_proj(f_sem).unsqueeze(1)  # [B,1,E]
        kv = self.local_proj(f_loc)                 # [B,E,H,W]
        kv = kv.flatten(2).transpose(1, 2)          # [B,HW,E]
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        z = self.norm1(q + attn_out)
        z2 = self.ffn(z)
        z = self.norm2(z + z2)
        return z.squeeze(1)  # [B,E]


class _ChannelFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VMamba2DPatchBlock(nn.Module):
    """Lightweight V-Mamba-style 2D patch mixer.

    This is not a full VMamba backbone. It is a small alignment-only spatial
    mixer designed for your IQA model:
      - input/output: patch tokens [B, N, C]
      - internally reshapes tokens to [B, C, H, W]
      - uses row/column/depthwise spatial selective mixing with a learned gate
      - keeps a residual path to avoid destroying quality-sensitive features

    Why use this only for alignment?
      Your full-fusion V-Mamba experiment improved AGIQA-3K alignment SROCC but
      damaged quality. This block is therefore applied only before the alignment
      patch-evidence branch, while the quality branch still uses the original
      patch tokens.
    """

    def __init__(self, dim: int = 256, dropout: float = 0.1, expansion: int = 2, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dw_row = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, padding), groups=dim)
        self.dw_col = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=dim)
        self.dw_local = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = _ChannelFFN(dim, expansion=expansion, dropout=dropout)

    def forward(self, x: torch.Tensor, hw: tuple[int, int] | None = None) -> torch.Tensor:
        # x: [B,N,C]
        b, n, c = x.shape
        if hw is None:
            h = int(round(math.sqrt(n)))
            w = n // h if h > 0 else n
            if h * w != n:
                # fallback: treat it as a 1D strip if N is not a square
                h, w = 1, n
        else:
            h, w = int(hw[0]), int(hw[1])
            if h * w != n:
                h = int(round(math.sqrt(n)))
                w = n // h if h > 0 else n
                if h * w != n:
                    h, w = 1, n

        y = self.norm(x)
        val, gate = self.in_proj(y).chunk(2, dim=-1)
        val2d = val.transpose(1, 2).reshape(b, c, h, w)

        # V-Mamba-style four-direction proxy: horizontal, vertical and local mixing.
        mix2d = self.dw_row(val2d) + self.dw_col(val2d) + self.dw_local(val2d)
        mix = mix2d.reshape(b, c, n).transpose(1, 2)
        mix = torch.sigmoid(gate) * mix
        x = x + self.drop(self.out_proj(mix))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class AlignOnlyVMambaFusion(nn.Module):
    """Stacked V-Mamba-style patch mixer for alignment branch only."""

    def __init__(self, dim: int = 256, depth: int = 1, dropout: float = 0.1, expansion: int = 2, kernel_size: int = 3, residual_scale: float = 0.5):
        super().__init__()
        self.blocks = nn.ModuleList([
            VMamba2DPatchBlock(dim=dim, dropout=dropout, expansion=expansion, kernel_size=kernel_size)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(dim)
        # Keep this small by default. It helps alignment while reducing the risk
        # of overwriting quality-sensitive patch tokens.
        self.residual_scale = float(residual_scale)

    def forward(self, patch_tokens: torch.Tensor, hw: tuple[int, int] | None = None) -> torch.Tensor:
        x = patch_tokens
        for blk in self.blocks:
            x = blk(x, hw=hw)
        x = self.norm(x)
        return patch_tokens + self.residual_scale * (x - patch_tokens)
