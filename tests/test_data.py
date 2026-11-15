import torch

from turbdiff.data import random_shifts


def test_shifts_are_circular_rolls_by_allowed_amounts():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, 16, 16, generator=g)
    out = random_shifts(x, period_y=4, generator=g)
    assert out.shape == x.shape
    for field, moved in zip(x, out, strict=True):
        found = False
        for sy in range(0, 16, 4):
            for sx in range(16):
                if torch.equal(torch.roll(field, (sy, sx), dims=(-2, -1)), moved):
                    found = True
        assert found


def test_shifts_use_every_allowed_amount():
    g = torch.Generator().manual_seed(1)
    base = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    seen = set()
    for _ in range(400):
        moved = random_shifts(base, 4, g)
        (iy, ix) = divmod(int(torch.argmin(moved).item()), 16)
        seen.add((iy % 4, iy // 4, ix))
    assert {s[0] for s in seen} == {0}
    assert len({s[1] for s in seen}) == 4
    assert len({s[2] for s in seen}) == 16
