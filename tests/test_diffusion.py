import torch
from torch import nn

from turbdiff import spectra
from turbdiff.diffusion import (
    EMA,
    Schedule,
    alpha_bar,
    ddim_sample,
    noisy,
    training_loss,
)
from turbdiff.gaussian import GaussianField
from turbdiff.unet import UNet


class GaussianDenoiser(nn.Module):
    """Exact noise predictor for data that is Gaussian with a given mode power."""

    def __init__(self, power, sched):
        super().__init__()
        n = power.shape[-1]
        self.lam = n**2 * power
        self.bars = sched.alphas_bar

    def forward(self, x, t):
        ab = self.bars[t[0]].item()
        gain = (1 - ab) ** 0.5 / (ab * self.lam + 1 - ab)
        spec = torch.fft.fft2(x.double(), norm="ortho") * gain
        return torch.fft.ifft2(spec, norm="ortho").real.float()


def test_cosine_schedule_starts_at_one_and_decreases():
    u = torch.linspace(0, 1, 1001, dtype=torch.float64)
    bar = alpha_bar(u)
    assert bar[0].item() == 1.0
    assert (bar[1:] < bar[:-1]).all()
    assert bar[-1].item() < 1e-12


def test_schedule_betas_are_valid():
    sched = Schedule()
    assert (sched.betas > 0).all() and (sched.betas <= 0.999).all()
    assert (sched.alphas_bar[1:] < sched.alphas_bar[:-1]).all()
    assert sched.alphas_bar[-1].item() < 1e-6


def test_forward_process_keeps_unit_variance_for_unit_variance_data():
    sched = Schedule()
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(64, 1, 16, 16, generator=g)
    noise = torch.randn(64, 1, 16, 16, generator=g)
    for t in (0, 300, 999):
        xt = noisy(sched, x0, torch.full((64,), t), noise)
        assert abs(xt.var().item() - 1.0) < 0.05


def test_ddim_is_deterministic_for_a_fixed_seed():
    model = UNet(base=8).eval()
    sched = Schedule()
    out = []
    for seed in (5, 5, 6):
        g = torch.Generator().manual_seed(seed)
        out.append(ddim_sample(model, sched, (2, 1, 16, 16), g, steps=5))
    assert torch.equal(out[0], out[1])
    assert not torch.equal(out[0], out[2])


def shell_power(p, n, kmax):
    idx = spectra.shell_index(n)
    return torch.stack([p[idx == k].sum() for k in range(1, kmax + 1)])


def test_ddim_with_the_exact_denoiser_recovers_the_gaussian_spectrum():
    n = 32
    ky, kx = spectra.wavenumbers(n)
    power = torch.exp(-(kx**2 + 2 * ky**2) / 40.0)
    power[0, 0] = 0.0
    power = power / power.sum()
    sched = Schedule()
    errors = {}
    for steps in (50, 250):
        g = torch.Generator().manual_seed(0)
        model = GaussianDenoiser(power, sched)
        out = ddim_sample(model, sched, (1000, 1, n, n), g, steps=steps)
        ratio = shell_power(spectra.mode_power(out[:, 0]), n, 6)
        errors[steps] = ((ratio / shell_power(power, n, 6)) - 1).abs().max().item()
    assert errors[250] < 0.05
    assert errors[50] > errors[250]


def test_replacement_ends_on_the_observed_values():
    model = UNet(base=8).eval()
    sched = Schedule()
    g = torch.Generator().manual_seed(1)
    mask = torch.rand(2, 1, 16, 16, generator=g) < 0.1
    values = torch.randn(2, 1, 16, 16, generator=g)
    for fresh in (True, False):
        out = ddim_sample(
            model,
            sched,
            (2, 1, 16, 16),
            g,
            steps=4,
            known=(mask, values),
            fresh_noise=fresh,
        )
        assert torch.equal(out[mask], values[mask])


def test_training_loss_falls_when_fitting_one_batch():
    torch.manual_seed(0)
    model = UNet(base=8)
    sched = Schedule()
    x0 = (
        GaussianField(torch.full((16, 16), 1 / 256))
        .sample(8, torch.Generator().manual_seed(0))[:, None]
        .float()
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(60):
        loss = training_loss(model, sched, x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < 0.9 * sum(losses[:10]) / 10


def test_ema_moves_towards_the_model():
    model = UNet(base=8)
    ema = EMA(model, decay=0.9)
    for p in model.parameters():
        p.data += 1.0
    before = next(ema.model.parameters()).clone()
    ema.update(model, step=1000)
    after = next(ema.model.parameters())
    target = next(model.parameters())
    assert (after - target).abs().mean() < (before - target).abs().mean()
