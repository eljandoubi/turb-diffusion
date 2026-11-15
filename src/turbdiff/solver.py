"""Pseudo-spectral solver for forced, damped 2D Navier-Stokes in vorticity form.

On the periodic square [0, 2 pi)^2 the vorticity w obeys

    dw/dt + u . grad(w) = nu lap(w) - mu w + amp cos(kf y),

with u = (dpsi/dy, -dpsi/dx) and lap(psi) = -w. Fields have shape (..., n, n)
with the y axis first and the x axis last, so w[..., j, i] sits at
(x, y) = (2 pi i / n, 2 pi j / n).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Params:
    """Flow and discretisation parameters."""

    n: int = 64
    nu: float = 1e-3
    mu: float = 0.1
    amp: float = 1.0
    kf: int = 4
    dt: float = 0.005


def retained(k2: torch.Tensor, n: int) -> torch.Tensor:
    """Modes kept by the 2/3 rule, those with 0 < round(|k|) <= n // 3."""
    return (k2 < (n // 3 + 0.5) ** 2) & (k2 > 0)


def grid(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinates (y, x) of the grid points, each of shape (n, n)."""
    x = 2 * torch.pi * torch.arange(n, dtype=torch.float64) / n
    return torch.meshgrid(x, x, indexing="ij")


class Solver:
    """Integrating-factor RK4 with the 2/3 rule, batched over leading axes."""

    def __init__(self, params: Params, device: str = "cpu"):
        n = params.n
        ky = torch.round(torch.fft.fftfreq(n, dtype=torch.float64) * n)[:, None]
        kx = torch.round(torch.fft.rfftfreq(n, dtype=torch.float64) * n)[None, :]
        k2 = kx**2 + ky**2
        self.params = params
        self.n = n
        self.dt = params.dt
        self.ikx = 1j * kx
        self.iky = 1j * ky
        self.inv_k2 = torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))
        self.mask = retained(k2, n).to(torch.float64)
        decay = -(params.nu * k2 + params.mu)
        self.e1 = torch.exp(decay * params.dt)
        self.e2 = torch.exp(decay * params.dt / 2)
        y = 2 * torch.pi * torch.arange(n, dtype=torch.float64) / n
        force = (params.amp * torch.cos(params.kf * y))[:, None].expand(n, n)
        self.force_hat = torch.fft.rfft2(force) * self.mask
        for name in ("ikx", "iky", "inv_k2", "mask", "e1", "e2", "force_hat"):
            setattr(self, name, getattr(self, name).to(device))

    def to_hat(self, omega: torch.Tensor) -> torch.Tensor:
        """Truncated half spectrum of a physical field."""
        return torch.fft.rfft2(omega.to(self.mask.device, torch.float64)) * self.mask

    def to_phys(self, w_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(w_hat, s=(self.n, self.n))

    def velocity(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        psi = w_hat * self.inv_k2
        u, v = torch.fft.irfft2(
            torch.stack((self.iky * psi, -self.ikx * psi)), s=(self.n, self.n)
        )
        return u, v

    def rhs(self, w_hat: torch.Tensor) -> torch.Tensor:
        """Advection and forcing, truncated to the retained modes."""
        psi = w_hat * self.inv_k2
        spec = torch.stack(
            (self.iky * psi, -self.ikx * psi, self.ikx * w_hat, self.iky * w_hat)
        )
        u, v, wx, wy = torch.fft.irfft2(spec, s=(self.n, self.n))
        advection = torch.fft.rfft2(u * wx + v * wy)
        return (self.force_hat - advection) * self.mask

    def step(self, w_hat: torch.Tensor) -> torch.Tensor:
        """One RK4 step with the linear terms treated by an integrating factor."""
        h, e1, e2 = self.dt, self.e1, self.e2
        a = self.rhs(w_hat)
        b = self.rhs(e2 * (w_hat + 0.5 * h * a))
        c = self.rhs(e2 * w_hat + 0.5 * h * b)
        d = self.rhs(e1 * w_hat + h * e2 * c)
        return e1 * w_hat + h / 6 * (e1 * a + 2 * e2 * (b + c) + d)

    @torch.no_grad()
    def advance(self, w_hat: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            w_hat = self.step(w_hat)
        return w_hat

    def evolve(self, omega: torch.Tensor, time: float) -> torch.Tensor:
        """Evolve physical fields for a given time and return physical fields."""
        steps = round(time / self.dt)
        return self.to_phys(self.advance(self.to_hat(omega), steps))


def random_vorticity(
    shape: tuple[int, ...], n: int, rms: float, k0: float, generator: torch.Generator
) -> torch.Tensor:
    """Smooth random fields with a Gaussian spectrum of width k0 and a given rms."""
    noise = torch.randn(*shape, n, n, dtype=torch.float64, generator=generator)
    ky = torch.round(torch.fft.fftfreq(n, dtype=torch.float64) * n)[:, None]
    kx = torch.round(torch.fft.fftfreq(n, dtype=torch.float64) * n)[None, :]
    keep = torch.exp(-((kx**2 + ky**2) / (2 * k0**2)))
    keep[0, 0] = 0.0
    field = torch.fft.ifft2(torch.fft.fft2(noise) * keep).real
    return field * rms / field.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
