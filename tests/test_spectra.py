import math

import pytest
import torch

from turbdiff import spectra
from turbdiff.solver import Params, Solver, grid, random_vorticity


def test_single_mode_puts_all_energy_in_its_shell():
    _, x = grid(64)
    amp, k = 3.0, 3
    e = spectra.energy_spectrum(amp * torch.cos(k * x))
    expected = amp**2 / (4 * k**2)
    assert e[k - 1].item() == pytest.approx(expected, rel=1e-12)
    e[k - 1] = 0.0
    assert e.abs().max() < 1e-20


def test_oblique_mode_lands_in_the_rounded_shell():
    y, x = grid(64)
    w = torch.cos(3 * x + 4 * y)
    e = spectra.energy_spectrum(w)
    assert e[4].item() == pytest.approx(0.25 / 25, rel=1e-12)
    assert e.sum().item() == pytest.approx(e[4].item(), rel=1e-12)


def test_energy_agrees_with_the_velocity_field():
    solver = Solver(Params())
    g = torch.Generator().manual_seed(0)
    w = solver.to_phys(solver.to_hat(random_vorticity((4,), 64, 3.0, 6.0, g)))
    u, v = solver.velocity(solver.to_hat(w))
    direct = 0.5 * (u**2 + v**2).mean(dim=(-2, -1))
    assert torch.allclose(spectra.energy(w), direct, rtol=1e-12)
    assert torch.allclose(spectra.energy_spectrum(w).sum(-1), direct, rtol=1e-12)


def test_enstrophy_spectrum_sums_to_half_the_mean_square():
    g = torch.Generator().manual_seed(1)
    w = random_vorticity((3,), 64, 2.0, 5.0, g)
    expected = 0.5 * w.pow(2).mean(dim=(-2, -1))
    assert torch.allclose(spectra.enstrophy(w), expected, rtol=1e-12)
    assert torch.allclose(spectra.enstrophy_spectrum(w).sum(-1), expected, rtol=1e-12)


def test_isotropic_average_keeps_shell_totals():
    g = torch.Generator().manual_seed(2)
    p = torch.rand(64, 64, dtype=torch.float64, generator=g)
    iso = spectra.isotropic(p)
    idx = spectra.shell_index(64)
    for k in (1, 5, 20):
        assert iso[idx == k].sum().item() == pytest.approx(p[idx == k].sum().item())
        assert iso[idx == k].std().item() < 1e-15


def test_outside_fraction_sees_modes_beyond_the_cutoff():
    _, x = grid(64)
    assert spectra.outside_fraction(torch.cos(5 * x)).item() < 1e-20
    assert spectra.outside_fraction(torch.cos(30 * x)).item() == pytest.approx(1.0)


def test_band_log_ratio_of_a_constant_factor():
    ref = torch.linspace(1.0, 2.0, 32, dtype=torch.float64)
    errors = spectra.band_errors(2.0 * ref, ref)
    for value in errors.values():
        assert value == pytest.approx(math.log(2.0))
