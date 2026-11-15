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


def start_index(sched: Schedule, min_alpha_bar: float) -> int:
    """Largest timestep whose signal level alpha_bar is at least min_alpha_bar."""
    return int((sched.alphas_bar >= min_alpha_bar).nonzero().max())


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    sched: Schedule,
    shape: tuple[int, ...],
    generator: torch.Generator,
    steps: int = 50,
    known: tuple[torch.Tensor, torch.Tensor] | None = None,
    fresh_noise: bool = True,
    min_alpha_bar: float = 1e-3,
) -> torch.Tensor:
    """Deterministic DDIM (eta = 0) from Gaussian noise.

    The chain starts at the last timestep with alpha_bar >= min_alpha_bar. At
    the end of the cosine schedule alpha_bar is 2e-9, and the clean estimate
    (x - sqrt(1 - ab) eps) / sqrt(ab) multiplies the error of the noise
    prediction by 2e4, which ruins the whole chain.

    known is an optional (mask, values) pair for conditioning by replacement:
    after every step the masked pixels are reset to the observed values noised
    to the current level, with fresh noise or with the noise the model predicted.
    """
    device = generator.device
    x = torch.randn(shape, generator=generator, device=device)
    first = start_index(sched, min_alpha_bar)
    ts = torch.linspace(first, 0, steps).round().long()
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


def sample_batches(
    model: nn.Module,
    sched: Schedule,
    count: int,
    n: int,
    device: str,
    steps: int,
    seed: int,
    chunk: int = 100,
    **kwargs,
) -> torch.Tensor:
    """DDIM samples of shape (count, n, n) in the units of the model input."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, count, chunk):
        shape = (min(chunk, count - i), 1, n, n)
        out.append(ddim_sample(model, sched, shape, g, steps=steps, **kwargs))
    return torch.cat(out)[:, 0].cpu().double()


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
