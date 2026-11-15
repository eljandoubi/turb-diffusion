import torch

from turbdiff import spectra
from turbdiff.diffusion import Schedule
from turbdiff.gaussian import GaussianField, condition, noise_prediction_error


def anisotropic_power(n=64, kmax=10):
    ky, kx = spectra.wavenumbers(n)
    k2 = kx**2 + ky**2
    power = torch.exp(-(kx**2 + 4 * ky**2) / 30.0)
    power = power * (k2 > 0) * (k2 <= kmax**2)
    return power / power.sum()


def test_samples_have_the_requested_power_per_mode():
    target = anisotropic_power()
    g = torch.Generator().manual_seed(0)
    x = GaussianField(target).sample(4000, g)
    est = spectra.mode_power(x)
    big = target > 1e-3 * target.max()
    assert ((est[big] / target[big]) - 1).abs().max() < 0.08
    assert abs(x.var().item() - 1.0) < 0.03


def test_fitted_power_is_symmetric_under_k_to_minus_k():
    target = anisotropic_power()
    g = torch.Generator().manual_seed(1)
    p = GaussianField.fit(GaussianField(target).sample(50, g)).power
    flipped = torch.roll(torch.flip(p, dims=(0, 1)), shifts=(1, 1), dims=(0, 1))
    assert torch.allclose(p, flipped, atol=1e-15)


def test_isotropic_variant_reproduces_the_shell_spectrum():
    target = anisotropic_power()
    g = torch.Generator().manual_seed(2)
    data = GaussianField(target).sample(3000, g)
    shell = GaussianField.fit(data, shell=True)
    ref = spectra.energy_spectrum(data).mean(0)
    got = spectra.energy_spectrum(shell.sample(3000, g)).mean(0)
    k = slice(0, 10)
    assert ((got[k] / ref[k]) - 1).abs().max() < 0.08
    assert shell.power.max() < GaussianField(target).power.max()


def test_kriging_interpolates_and_improves_with_more_points():
    field = GaussianField(anisotropic_power())
    g = torch.Generator().manual_seed(3)
    truth = field.sample(1, g)[0]
    errors = []
    for m in (40, 160, 640):
        index = torch.randperm(64 * 64, generator=g)[:m]
        est = condition(field, index, truth)
        seen = (est.reshape(-1)[index] - truth.reshape(-1)[index]).abs().max()
        assert seen < 1e-5 * truth.abs().max()
        errors.append((est - truth).norm() / truth.norm())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-2


def test_conditional_sample_matches_the_observations():
    field = GaussianField(anisotropic_power())
    g = torch.Generator().manual_seed(4)
    truth, prior = field.sample(2, g)
    index = torch.randperm(64 * 64, generator=g)[:100]
    sample = condition(field, index, truth, prior=prior)
    diff = sample.reshape(-1)[index] - truth.reshape(-1)[index]
    assert diff.abs().max() < 1e-5 * truth.abs().max()
    assert (sample - truth).norm() < (prior - truth).norm()


def test_gaussian_noise_prediction_error_matches_its_formula():
    power = anisotropic_power()
    sched = Schedule()
    g = torch.Generator().manual_seed(5)
    x0 = GaussianField(power).sample(600, g)
    noise = torch.randn(600, 64, 64, generator=g, dtype=torch.float64)
    t = torch.full((600,), 300)
    ab = sched.alphas_bar[300].item()
    lam = 64**2 * power
    expected = (ab * lam / (ab * lam + 1 - ab)).mean().item()
    got = noise_prediction_error(power, sched, x0, t, noise)
    assert abs(got / expected - 1) < 0.03
