"""Statistics that compare sets of vorticity fields of shape (N, n, n)."""

import numpy as np
import torch

SEPARATIONS = (1, 2, 4, 8, 16)
ORDERS = (2, 4)


def moments(x: torch.Tensor) -> dict[str, float]:
    """Skewness and excess kurtosis of the values pooled over all fields."""
    v = x.double().reshape(-1)
    z = (v - v.mean()) / v.std(unbiased=False)
    return {
        "skewness": z.pow(3).mean().item(),
        "excess_kurtosis": z.pow(4).mean().item() - 3.0,
    }


def ks_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov distance between the pooled values."""
    a = np.sort(a.double().reshape(-1).numpy())
    b = np.sort(b.double().reshape(-1).numpy())
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def histogram(x: torch.Tensor, scale: float, bins: int = 100, limit: float = 10.0):
    """Density of x / scale on [-limit, limit], returned as (centres, density)."""
    edges = torch.linspace(-limit, limit, bins + 1, dtype=torch.float64)
    counts = torch.histc((x.double() / scale).reshape(-1), bins, -limit, limit)
    density = counts / (counts.sum() * (edges[1] - edges[0]))
    return ((edges[1:] + edges[:-1]) / 2).tolist(), density.tolist()


def structure_functions(x: torch.Tensor) -> dict[str, dict[str, list[float]]]:
    """Moments of |w(r) - w(0)| of orders 2 and 4 along x and y, per separation."""
    x = x.double()
    out: dict[str, dict[str, list[float]]] = {}
    for order in ORDERS:
        out[f"S{order}"] = {"x": [], "y": []}
        for axis, name in ((-1, "x"), (-2, "y")):
            for r in SEPARATIONS:
                inc = torch.roll(x, -r, dims=axis) - x
                out[f"S{order}"][name].append(inc.abs().pow(order).mean().item())
    return out


def structure_error(sf: dict, ref: dict) -> dict[str, float]:
    """Mean |log(S / S_ref)| over separations and both directions, per order."""
    out = {}
    for order in sf:
        diffs = [
            abs(np.log(a / b))
            for d in ("x", "y")
            for a, b in zip(sf[order][d], ref[order][d], strict=True)
        ]
        out[order] = float(np.mean(diffs))
    return out


def forced_mode_coherence(x: torch.Tensor, kf: int) -> float:
    """|mean| over fields of the forced Fourier mode divided by its rms.

    It is 1 when every field carries the same phase of the forcing and near 0
    when the phase is random.
    """
    coeff = torch.fft.fft2(x.double(), norm="forward")[:, kf, 0]
    return (coeff.mean().abs() / coeff.abs().pow(2).mean().sqrt()).item()


def nearest_distance(
    fields: torch.Tensor, train: torch.Tensor, period_y: int, chunk: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rms pixel distance from each field to its nearest training field.

    The first result compares the fields as they are, the second also allows
    the shifts under which the flow statistics are invariant (any shift along x,
    shifts along y by multiples of period_y).
    """
    n = fields.shape[-1]
    a, b = fields.float(), train.to(fields.device).float()
    na, nb = a.pow(2).mean(dim=(-2, -1)), b.pow(2).mean(dim=(-2, -1))
    plain = a.reshape(len(a), -1) @ b.reshape(len(b), -1).T / n**2
    shifted = []
    bf = torch.fft.rfft2(b).conj()[None]
    for i in range(0, len(a), chunk):
        corr = torch.fft.irfft2(
            torch.fft.rfft2(a[i : i + chunk])[:, None] * bf, s=(n, n)
        )
        shifted.append(corr[..., ::period_y, :].amax(dim=(-2, -1)) / n**2)
    shifted = torch.cat(shifted)
    out = []
    for inner in (plain, shifted):
        d2 = na[:, None] + nb[None, :] - 2 * inner
        out.append(d2.amin(dim=1).clamp(min=0).sqrt())
    return out[0], out[1]
