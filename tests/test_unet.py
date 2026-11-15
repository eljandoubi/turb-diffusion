import torch

from turbdiff.unet import UNet, conv, count_parameters


def perturbed(model):
    g = torch.Generator().manual_seed(0)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.1
    return model


def test_periodic_convolution_commutes_with_any_shift():
    layer = conv(2, 3).double()
    x = torch.randn(1, 2, 16, 16, dtype=torch.float64)
    for shift in [(1, 0), (0, 5), (7, 11)]:
        moved = layer(torch.roll(x, shift, dims=(-2, -1)))
        assert torch.allclose(
            moved, torch.roll(layer(x), shift, dims=(-2, -1)), atol=1e-12
        )


def test_strided_periodic_convolution_commutes_with_even_shifts():
    layer = conv(2, 3, stride=2).double()
    x = torch.randn(1, 2, 16, 16, dtype=torch.float64)
    out = layer(x)
    for shift in [(2, 0), (0, 6), (4, 10)]:
        moved = layer(torch.roll(x, shift, dims=(-2, -1)))
        half = (shift[0] // 2, shift[1] // 2)
        assert torch.allclose(moved, torch.roll(out, half, dims=(-2, -1)), atol=1e-12)


def test_output_has_the_input_shape_and_starts_at_zero():
    model = UNet(base=8)
    x = torch.randn(2, 1, 32, 32)
    out = model(x, torch.tensor([3, 700]))
    assert out.shape == x.shape
    assert out.abs().max() == 0


def test_default_network_has_a_few_million_parameters():
    assert 3e6 < count_parameters(UNet()) < 5e6


def test_unet_commutes_with_shifts_by_the_total_stride():
    model = perturbed(UNet(base=8)).double().eval()
    x = torch.randn(2, 1, 32, 32, dtype=torch.float64)
    t = torch.tensor([10, 500])
    ref = model(x, t)
    for shift in [(8, 0), (0, 16), (24, 8)]:
        moved = model(torch.roll(x, shift, dims=(-2, -1)), t)
        assert torch.allclose(moved, torch.roll(ref, shift, dims=(-2, -1)), atol=1e-9)


def test_unet_does_not_commute_with_a_single_pixel_shift():
    model = perturbed(UNet(base=8)).double().eval()
    x = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    t = torch.tensor([10])
    moved = model(torch.roll(x, (1, 0), dims=(-2, -1)), t)
    assert not torch.allclose(
        moved, torch.roll(model(x, t), (1, 0), (-2, -1)), atol=1e-6
    )
