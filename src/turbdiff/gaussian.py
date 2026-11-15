"""Gaussian random field baselines and Gaussian conditioning (kriging)."""

import torch
from torch import nn

from turbdiff import spectra
from turbdiff.diffusion import Schedule


class GaussianDenoiser(nn.Module):
    """Exact noise predictor for Gaussian data with a given power per mode.

    It stands in for a perfectly trained network, so that sampling with it shows
    what the sampler alone does to a Gaussian field.
    """

    def __init__(self, power: torch.Tensor, sched: Schedule):
        super().__init__()
        self.lam = power.shape[-1] ** 2 * power.double()
        self.bars = sched.alphas_bar

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lam = self.lam.to(x.device)
        ab = self.bars[int(t[0])].item()
        gain = (1 - ab) ** 0.5 / (ab * lam + 1 - ab)
        spec = torch.fft.fft2(x.double(), norm="ortho") * gain
        return torch.fft.ifft2(spec, norm="ortho").real.to(x.dtype)


class GaussianField:
    """Zero-mean stationary Gaussian field with a given power per Fourier mode."""

    def __init__(self, power: torch.Tensor):
        self.power = power.to(torch.float64)
        self.n = power.shape[-1]

    @classmethod
    def fit(cls, x: torch.Tensor, shell: bool = False) -> "GaussianField":
        """Match the mean power of every mode of x, or of every shell if shell."""
        p = spectra.mode_power(x)
        return cls(spectra.isotropic(p) if shell else p)

    def sample(self, count: int, generator: torch.Generator) -> torch.Tensor:
        n = self.n
        white = torch.randn(count, n, n, dtype=torch.float64, generator=generator)
        spec = torch.fft.fft2(white, norm="ortho") * (n * self.power.sqrt())
        return torch.fft.ifft2(spec, norm="ortho").real

    def covariance(self) -> torch.Tensor:
        """Covariance of two points as a function of their offset (dy, dx)."""
        return (self.n**2 * torch.fft.ifft2(self.power)).real


def noise_prediction_error(
    power: torch.Tensor,
    sched: Schedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> float:
    """Mean squared error of the exact Gaussian noise predictor on fields x0.

    x0 and noise have shape (B, n, n), t holds one timestep per field.
    """
    lam = power.shape[-1] ** 2 * power.double()
    ab = sched.alphas_bar[t][:, None, None]
    xt = ab.sqrt() * x0.double() + (1 - ab).sqrt() * noise.double()
    gain = (1 - ab).sqrt() / (ab * lam + 1 - ab)
    spec = torch.fft.fft2(xt, norm="ortho") * gain
    pred = torch.fft.ifft2(spec, norm="ortho").real
    return ((pred - noise.double()) ** 2).mean().item()


def krige(
    cov: torch.Tensor, index: torch.Tensor, values: torch.Tensor, jitter: float = 1e-9
) -> torch.Tensor:
    """Posterior mean of a zero-mean Gaussian field given exact values at points.

    cov is the offset covariance of GaussianField, index holds flat pixel numbers
    (m,) and values the observations (m,). Returns the field of shape (n, n).
    """
    n = cov.shape[-1]
    iy, ix = index // n, index % n
    c_ss = cov[(iy[:, None] - iy[None, :]) % n, (ix[:, None] - ix[None, :]) % n]
    c_ss = c_ss + jitter * cov[0, 0] * torch.eye(len(index), dtype=cov.dtype)
    weights = torch.linalg.solve(c_ss, values.to(cov.dtype))
    sparse = torch.zeros(n * n, dtype=cov.dtype)
    sparse[index] = weights
    conv = torch.fft.ifft2(torch.fft.fft2(sparse.reshape(n, n)) * torch.fft.fft2(cov))
    return conv.real


def condition(
    field: GaussianField,
    index: torch.Tensor,
    truth: torch.Tensor,
    prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Kriging estimate of truth from its values at index.

    With a prior draw from the field the result is a posterior sample instead
    of the posterior mean (the draw is corrected to match the observations).
    """
    cov = field.covariance()
    flat = truth.reshape(-1)
    if prior is None:
        return krige(cov, index, flat[index])
    return prior + krige(cov, index, flat[index] - prior.reshape(-1)[index])
