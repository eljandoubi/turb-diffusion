"""Periodic UNet: circular padding everywhere, so it commutes with grid shifts."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of timesteps t of shape (B,), returned as (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=1)


def conv(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, padding_mode="circular")


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int, dropout: float, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = conv(cin, cout)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = conv(cout, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    """Self-attention over all positions, without a position code."""

    def __init__(self, ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = (t.transpose(-1, -2) for t in qkv.unbind(1))
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Noise predictor for fields of shape (B, 1, n, n), n divisible by 2 ** depth."""

    def __init__(
        self,
        base: int = 48,
        mult: tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        chs = [base * m for m in mult]
        tdim = 4 * base

        def block(cin: int, cout: int) -> ResBlock:
            return ResBlock(cin, cout, tdim, dropout)

        self.base = base
        self.stride = 2 ** (len(chs) - 1)
        self.time = nn.Sequential(
            nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )
        self.inp = conv(1, base)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = base
        for i, ch in enumerate(chs):
            self.enc.append(nn.ModuleList([block(c, ch), block(ch, ch)]))
            c = ch
            if i < len(chs) - 1:
                self.down.append(conv(c, c, stride=2))
        self.mid1 = block(c, c)
        self.attn = Attention(c)
        self.mid2 = block(c, c)
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in reversed(range(len(chs))):
            self.dec.append(
                nn.ModuleList([block(2 * chs[i], chs[i]), block(chs[i], chs[i])])
            )
            if i > 0:
                self.up.append(conv(chs[i], chs[i - 1]))
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out = conv(chs[0], 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time(timestep_embedding(t, self.base).to(x.dtype))
        h = self.inp(x)
        skips = []
        for i, blocks in enumerate(self.enc):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        h = self.mid2(self.attn(self.mid1(h, temb)), temb)
        for i, blocks in enumerate(self.dec):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, temb)
            if i < len(self.up):
                h = self.up[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.out_norm(h)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
