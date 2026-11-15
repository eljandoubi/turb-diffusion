"""Shell spectra and related diagnostics for periodic fields on an n x n grid."""

import functools

import torch

from turbdiff.solver import retained

BANDS = {"low": (1, 4), "mid": (5, 10), "high": (11, 21)}


def wavenumbers(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer wavenumber grids (ky, kx) of the full spectrum, as (n, 1), (1, n)."""
    k = torch.round(torch.fft.fftfreq(n, dtype=torch.float64) * n)
    return k[:, None], k[None, :]


@functools.lru_cache(maxsize=None)
def shell_index(n: int) -> torch.Tensor:
    """Shell number round(|k|) of every mode of the full spectrum."""
    ky, kx = wavenumbers(n)
    return torch.round(torch.sqrt(kx**2 + ky**2)).long()


@functools.lru_cache(maxsize=None)
def _shell_matrix(n: int) -> torch.Tensor:
    """One-hot matrix (n * n, n // 2) mapping modes to the shells 1 .. n // 2."""
    idx = shell_index(n).reshape(-1)
    out = torch.zeros(n * n, n // 2, dtype=torch.float64)
    for k in range(1, n // 2 + 1):
        out[idx == k, k - 1] = 1.0
    return out


@functools.lru_cache(maxsize=None)
def _inv_k2(n: int) -> torch.Tensor:
    ky, kx = wavenumbers(n)
    k2 = kx**2 + ky**2
    return torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))


def power(x: torch.Tensor) -> torch.Tensor:
    """Power of every mode, normalised so that its sum is the mean square of x."""
    return torch.fft.fft2(x.to(torch.float64), norm="forward").abs() ** 2


def mode_power(x: torch.Tensor) -> torch.Tensor:
    """Power per mode averaged over all leading axes of x."""
    n = x.shape[-1]
    return power(x).reshape(-1, n, n).mean(dim=0)


def energy(x: torch.Tensor) -> torch.Tensor:
    """Kinetic energy 0.5 <u^2 + v^2> of vorticity fields, one value per field."""
    n = x.shape[-1]
    return 0.5 * (power(x) * _inv_k2(n)).sum(dim=(-2, -1))


def enstrophy(x: torch.Tensor) -> torch.Tensor:
    """Enstrophy 0.5 <w^2> of vorticity fields, one value per field."""
    return 0.5 * power(x).sum(dim=(-2, -1))


def energy_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Shell-summed energy spectrum E(k) for k = 1 .. n // 2, per field."""
    n = x.shape[-1]
    weighted = (power(x) * _inv_k2(n)).reshape(*x.shape[:-2], n * n)
    return 0.5 * weighted @ _shell_matrix(n)


def enstrophy_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Shell-summed enstrophy spectrum Z(k) for k = 1 .. n // 2, per field."""
    n = x.shape[-1]
    return 0.5 * power(x).reshape(*x.shape[:-2], n * n) @ _shell_matrix(n)


def isotropic(p: torch.Tensor) -> torch.Tensor:
    """Replace the power of each mode by the mean power of its shell."""
    n = p.shape[-1]
    idx = shell_index(n).reshape(-1)
    total = torch.bincount(idx, weights=p.reshape(-1))
    count = torch.bincount(idx)
    return (total / count)[idx].reshape(n, n)


def outside_fraction(x: torch.Tensor) -> torch.Tensor:
    """Fraction of the mean square held by modes the solver truncates."""
    n = x.shape[-1]
    ky, kx = wavenumbers(n)
    kept = retained(kx**2 + ky**2, n)
    p = power(x)
    return (p * ~kept).sum(dim=(-2, -1)) / p.sum(dim=(-2, -1))


def band_log_ratio(e: torch.Tensor, ref: torch.Tensor, lo: int, hi: int) -> float:
    """Mean |log(e / ref)| over the shells lo .. hi of two spectra E(1 .. n // 2)."""
    sl = slice(lo - 1, hi)
    return (torch.log(e[sl]) - torch.log(ref[sl])).abs().mean().item()


def band_errors(e: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """band_log_ratio over the low, mid and high bands plus all of them."""
    out = {name: band_log_ratio(e, ref, lo, hi) for name, (lo, hi) in BANDS.items()}
    out["all"] = band_log_ratio(e, ref, 1, BANDS["high"][1])
    return out
