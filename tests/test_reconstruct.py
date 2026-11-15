import math

import pytest
import torch

from turbdiff.reconstruct import point_mask, random_points, relative_l2, spectrum_error
from turbdiff.solver import random_vorticity


def test_random_points_are_distinct_and_repeatable():
    a = random_points(64, 100, torch.Generator().manual_seed(0))
    b = random_points(64, 100, torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
    assert len(set(a.tolist())) == 100
    assert point_mask(64, a).sum().item() == 100


def test_relative_l2_of_exact_and_zero_estimates():
    truth = torch.randn(3, 16, 16, generator=torch.Generator().manual_seed(1))
    assert relative_l2(truth, truth).abs().max() == 0
    assert torch.allclose(relative_l2(torch.zeros_like(truth), truth), torch.ones(3))


def test_spectrum_error_of_scaled_fields():
    g = torch.Generator().manual_seed(2)
    w = random_vorticity((2,), 64, 3.0, 5.0, g)
    assert spectrum_error(w, w).abs().max() < 1e-12
    assert spectrum_error(2.0 * w, w) == pytest.approx(
        torch.full((2,), math.log(4.0), dtype=torch.float64), abs=1e-9
    )
