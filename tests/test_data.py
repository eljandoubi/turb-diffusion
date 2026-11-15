import math

import pytest
import torch

from turbdiff.data import (
    half_turn,
    random_symmetries,
    reflect_x,
    shift_reflect,
    symmetry_cosets,
)
from turbdiff.solver import Params, Solver, grid, random_vorticity


def test_reflections_and_turns_square_to_shifts():
    x = torch.randn(1, 1, 16, 16, generator=torch.Generator().manual_seed(0))
    assert torch.equal(reflect_x(reflect_x(x)), x)
    assert torch.equal(half_turn(half_turn(x)), x)
    twice = shift_reflect(shift_reflect(x, 2), 2)
    assert torch.equal(twice, torch.roll(x, -4, dims=-2))


def test_maps_act_on_coordinates_as_described():
    y, x = grid(32)
    w = torch.sin(2 * x + y)
    assert torch.allclose(reflect_x(w), torch.sin(-2 * x + y))
    assert torch.allclose(half_turn(w), torch.sin(-2 * x - y))
    expected = -torch.sin(-2 * x + y + math.pi / 4)
    assert torch.allclose(shift_reflect(w, 4), expected)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_solver_commutes_with_the_symmetries_of_the_flow(index):
    params = Params()
    solver = Solver(params)
    fn = symmetry_cosets(params.n // params.kf)[index]
    g = torch.Generator().manual_seed(index)
    w = random_vorticity((2,), params.n, 2.0, 4.0, g)
    moved_after = fn(solver.evolve(w, 0.5))
    moved_before = solver.evolve(fn(w), 0.5)
    assert (moved_after - moved_before).abs().max() < 1e-9 * w.abs().max()


def test_random_symmetries_stay_in_the_group_and_cover_it():
    n, period = 16, 4
    base = torch.arange(1, n * n + 1, dtype=torch.float32).reshape(1, 1, n, n)
    group = set()
    for fn in symmetry_cosets(period):
        for sy in range(0, n, period):
            for sx in range(n):
                moved = torch.roll(fn(base), (sy, sx), dims=(-2, -1))
                group.add(tuple(moved.flatten().tolist()))
    assert len(group) == 4 * (n // period) * n
    g = torch.Generator().manual_seed(1)
    out = random_symmetries(base.expand(8000, 1, n, n), period, g)
    seen = {tuple(o.flatten().tolist()) for o in out}
    assert seen == group
