import math

import pytest
import torch

from turbdiff import diagnostics as dg
from turbdiff.solver import grid


def test_moments_of_gaussian_and_exponential_samples():
    g = torch.Generator().manual_seed(0)
    gauss = dg.moments(torch.randn(200000, generator=g))
    assert abs(gauss["skewness"]) < 0.02 and abs(gauss["excess_kurtosis"]) < 0.05
    expo = dg.moments(-torch.log(torch.rand(400000, generator=g)))
    assert expo["skewness"] == pytest.approx(2.0, abs=0.1)
    assert expo["excess_kurtosis"] == pytest.approx(6.0, abs=0.6)


def test_ks_distance_separates_distributions():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(50000, generator=g), torch.randn(50000, generator=g)
    assert dg.ks_distance(a, b) < 0.02
    assert dg.ks_distance(a, 1.0 + b) > 0.3


def test_histogram_integrates_to_one():
    g = torch.Generator().manual_seed(2)
    centres, density = dg.histogram(torch.randn(100000, generator=g), 1.0)
    width = centres[1] - centres[0]
    assert sum(density) * width == pytest.approx(1.0, abs=1e-9)


def test_structure_functions_of_a_cosine():
    _, x = grid(64)
    sf = dg.structure_functions(torch.cos(x)[None])
    for r, s2, s4 in zip(dg.SEPARATIONS, sf["S2"]["x"], sf["S4"]["x"], strict=True):
        delta = 2 * math.pi * r / 64
        assert s2 == pytest.approx(1 - math.cos(delta), rel=1e-9)
        assert s4 == pytest.approx(6 * math.sin(delta / 2) ** 4, rel=1e-9)
    assert max(sf["S2"]["y"]) < 1e-20


def test_structure_error_of_identical_and_scaled_fields():
    g = torch.Generator().manual_seed(3)
    sf = dg.structure_functions(torch.randn(4, 32, 32, generator=g))
    assert all(v == 0.0 for v in dg.structure_error(sf, sf).values())
    scaled = dg.structure_functions(2.0 * torch.randn(4, 32, 32, generator=g))
    err = dg.structure_error(scaled, sf)
    assert err["S2"] == pytest.approx(math.log(4.0), abs=0.2)
    assert err["S4"] == pytest.approx(math.log(16.0), abs=0.4)


def test_forced_mode_coherence_with_fixed_and_random_phase():
    y, _ = grid(64)
    g = torch.Generator().manual_seed(4)
    phases = torch.rand(400, 1, 1, generator=g, dtype=torch.float64) * 2 * math.pi
    assert dg.forced_mode_coherence(torch.cos(4 * y)[None] + 0 * phases, 4) == (
        pytest.approx(1.0)
    )
    assert dg.forced_mode_coherence(torch.cos(4 * y[None] + phases), 4) < 0.15


def test_nearest_distance_sees_copies_up_to_the_allowed_shifts():
    g = torch.Generator().manual_seed(5)
    train = torch.randn(20, 32, 32, generator=g)
    queries = torch.stack(
        [
            train[3],
            torch.roll(train[7], (8, 5), dims=(-2, -1)),
            torch.roll(train[9], (1, 0), dims=(-2, -1)),
            torch.randn(32, 32, generator=g),
        ]
    )
    plain, shifted = dg.nearest_distance(queries, train, period_y=8)
    assert plain[0] < 1e-3 and shifted[0] < 1e-3
    assert plain[1] > 0.5 and shifted[1] < 1e-3
    assert shifted[2] > 0.5
    assert shifted[3] > 0.5
    assert (shifted <= plain + 1e-6).all()
