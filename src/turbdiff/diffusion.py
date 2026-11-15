"""DDPM with a cosine schedule, epsilon prediction and a deterministic DDIM sampler."""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


def alpha_bar(u: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule: cumulative signal level at u in [0, 1], equal to 1 at 0."""

    def f(v):
        return torch.cos((v + s) / (1 + s) * math.pi / 2) ** 2

    return f(u) / f(torch.zeros_like(u))


class Schedule:
    """Noise levels of the forward process; index i is the state after i + 1 steps."""

    def __init__(self, steps: int = 1000, max_beta: float = 0.999):
        u = torch.arange(steps + 1, dtype=torch.float64) / steps
        bar = alpha_bar(u)
        self.steps = steps
        self.betas = (1 - bar[1:] / bar[:-1]).clamp(max=max_beta)
        self.alphas_bar = torch.cumprod(1 - self.betas, dim=0)


def noisy(sched: Schedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    """Forward process sample x_t = sqrt(ab) x0 + sqrt(1 - ab) noise."""
    ab = sched.alphas_bar.to(x0.device)[t].to(x0.dtype)[:, None, None, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def training_loss(model: nn.Module, sched: Schedule, x0: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the noise prediction at uniformly drawn timesteps."""
    t = torch.randint(0, sched.steps, (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    pred = model(noisy(sched, x0, t, noise), t)
    return F.mse_loss(pred.float(), noise)


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    sched: Schedule,
    shape: tuple[int, ...],
    generator: torch.Generator,
    steps: int = 50,
    known: tuple[torch.Tensor, torch.Tensor] | None = None,
    fresh_noise: bool = True,
) -> torch.Tensor:
    """Deterministic DDIM (eta = 0) from Gaussian noise.

    known is an optional (mask, values) pair for conditioning by replacement:
    after every step the masked pixels are reset to the observed values noised
    to the current level, with fresh noise or with the noise the model predicted.
    """
    device = generator.device
    x = torch.randn(shape, generator=generator, device=device)
    ts = torch.linspace(sched.steps - 1, 0, steps).round().long()
    bars = sched.alphas_bar
    for i, t in enumerate(ts.tolist()):
        ab = bars[t].item()
        ab_next = bars[ts[i + 1]].item() if i + 1 < steps else 1.0
        eps = model(x, torch.full((shape[0],), t, device=device)).float()
        x0 = (x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
        x = math.sqrt(ab_next) * x0 + math.sqrt(1 - ab_next) * eps
        if known is not None:
            mask, values = known
            level = eps
            if fresh_noise:
                level = torch.randn(shape, generator=generator, device=device)
            pinned = math.sqrt(ab_next) * values + math.sqrt(1 - ab_next) * level
            x = torch.where(mask, pinned, x)
    return x


class EMA:
    """Exponential moving average of the weights, with a warm-up of the decay."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for ema_p, p in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p, 1 - d)
