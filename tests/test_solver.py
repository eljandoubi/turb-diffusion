import math

import pytest
import torch

from turbdiff.solver import Params, Solver, grid, random_vorticity


def kinetic_energy(solver, omega):
    u, v = solver.velocity(solver.to_hat(omega))
    return 0.5 * (u**2 + v**2).mean(dim=(-2, -1))


def test_taylor_green_is_steady_without_viscosity():
    solver = Solver(Params(nu=0.0, mu=0.0, amp=0.0))
    y, x = grid(solver.n)
    w0 = 2 * torch.sin(x) * torch.sin(y)
    w = solver.evolve(w0, 0.5)
    assert (w - w0).abs().max() < 1e-12


@pytest.mark.parametrize("nu,mu", [(0.02, 0.0), (0.02, 0.1)])
def test_taylor_green_decays_at_the_exact_rate(nu, mu):
    solver = Solver(Params(nu=nu, mu=mu, amp=0.0))
    y, x = grid(solver.n)
    w0 = 2 * torch.sin(x) * torch.sin(y)
    w = solver.evolve(w0, 1.0)
    exact = w0 * math.exp(-(2 * nu + mu))
    assert (w - exact).abs().max() / exact.abs().max() < 1e-10


def test_zero_vorticity_stays_zero():
    solver = Solver(Params(amp=0.0))
    w = solver.evolve(torch.zeros(3, 64, 64), 0.1)
    assert w.abs().max() == 0


def test_laminar_kolmogorov_flow_is_steady():
    p = Params(nu=1e-3, mu=0.1, amp=0.5, kf=4)
    solver = Solver(p)
    y, _ = grid(solver.n)
    w0 = p.amp / (p.nu * p.kf**2 + p.mu) * torch.cos(p.kf * y)
    w = solver.evolve(w0, 1.0)
    assert (w - w0).abs().max() / w0.abs().max() < 1e-10


def test_advection_matches_an_analytic_example():
    solver = Solver(Params(amp=0.0))
    y, x = grid(solver.n)
    w = torch.cos(x) + torch.cos(2 * y)
    got = solver.to_phys(solver.rhs(solver.to_hat(w)))
    assert (got - 1.5 * torch.sin(x) * torch.sin(2 * y)).abs().max() < 1e-12


def test_advection_conserves_energy_and_enstrophy():
    solver = Solver(Params(amp=0.0))
    g = torch.Generator().manual_seed(0)
    w_hat = solver.to_hat(random_vorticity((3,), 64, 5.0, 8.0, g))
    w = solver.to_phys(w_hat)
    psi = solver.to_phys(w_hat * solver.inv_k2)
    adv = solver.to_phys(solver.rhs(w_hat))
    for field in (psi, w):
        rate = (field * adv).mean(dim=(-2, -1))
        scale = field.pow(2).mean(dim=(-2, -1)).sqrt() * adv.pow(2).mean().sqrt()
        assert (rate.abs() / scale).max() < 1e-12


def test_energy_is_conserved_in_the_inviscid_undriven_limit():
    solver = Solver(Params(nu=0.0, mu=0.0, amp=0.0))
    g = torch.Generator().manual_seed(1)
    w0 = solver.to_phys(solver.to_hat(random_vorticity((2,), 64, 2.0, 4.0, g)))
    e0 = kinetic_energy(solver, w0)
    e1 = kinetic_energy(solver, solver.evolve(w0, 0.5))
    assert ((e1 - e0).abs() / e0).max() < 1e-8


def test_batches_are_solved_independently():
    solver = Solver(Params())
    g = torch.Generator().manual_seed(2)
    w = random_vorticity((2,), 64, 3.0, 5.0, g)
    together = solver.evolve(w, 0.1)
    alone = torch.stack([solver.evolve(w[i], 0.1) for i in range(2)])
    assert (together - alone).abs().max() < 1e-12


def test_time_stepping_has_fourth_order_accuracy():
    g = torch.Generator().manual_seed(3)
    w0 = random_vorticity((1,), 64, 3.0, 3.0, g)

    def run(dt):
        return Solver(Params(dt=dt)).evolve(w0, 0.32)

    ref = run(0.00125)
    err = [(run(dt) - ref).pow(2).sum().sqrt() for dt in (0.02, 0.01)]
    assert 10 < err[0] / err[1] < 24
