"""Helpers for reconstructing fields from values at a few random grid points."""

import torch

from turbdiff import spectra
from turbdiff.solver import Params

KMAX = Params().n // 3


def random_points(n: int, count: int, generator: torch.Generator) -> torch.Tensor:
    """Flat indices of count distinct random grid points of an n x n grid."""
    return torch.randperm(n * n, generator=generator)[:count]


def point_mask(n: int, index: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(n * n, dtype=torch.bool)
    mask[index] = True
    return mask.reshape(n, n)


def relative_l2(est: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Relative L2 error of each field of (N, n, n)."""
    diff = (est - truth).flatten(1).norm(dim=1)
    return diff / truth.flatten(1).norm(dim=1)


def spectrum_error(est: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Mean |log(E_est / E_truth)| over shells 1 .. KMAX, for each field."""
    e, t = spectra.energy_spectrum(est), spectra.energy_spectrum(truth)
    return (torch.log(e[:, :KMAX]) - torch.log(t[:, :KMAX])).abs().mean(dim=1)
