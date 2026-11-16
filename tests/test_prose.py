"""The figures that the README states in sentences are recomputed from results/."""

import json
import math
from pathlib import Path

from turbdiff.diffusion import Schedule

ROOT = Path(__file__).resolve().parents[1]
CHANGES = ("energy_relative_change", "enstrophy_relative_change")


def results(name):
    return json.loads((ROOT / "results" / name).read_text())


def passage(anchor):
    """The one paragraph or list item of the README that holds anchor."""
    found, run = [], []
    for line in [*(ROOT / "README.md").read_text().splitlines(), ""]:
        if not line.strip() or line.startswith("- "):
            if run:
                found.append(" ".join(run))
                run = []
        if line.strip():
            run.append(line.strip())
    found = [text for text in found if anchor in text]
    assert len(found) == 1, (anchor, len(found))
    return found[0]


def carries(text, *phrases):
    assert [p for p in phrases if p not in text] == []


def relaxation(ratio, shells):
    return sum(abs(math.log(r)) for r in ratio[:shells]) / shells


def spectrum_ratios():
    sources = results("diagnostics.json")["sources"]
    real = sources["real_test"]["spectrum"][:21]
    return [
        [a / b for a, b in zip(sources[name]["spectrum"][:21], real, strict=True)]
        for name in ("diffusion", "diffusion_plain")
    ]


def test_enstrophy_figures():
    s = results("diagnostics.json")["sources"]
    e = {name: f"{v['enstrophy_mean']:.3g}" for name, v in s.items()}
    carries(
        passage("Where the spectrum decides"),
        f"enstrophy {e['g_mode']} and {e['g_shell']} against {e['real_test']})",
        f"its mean enstrophy is {e['diffusion']} and {e['diffusion_plain']} "
        f"against {e['real_test']} for held-out fields",
    )


def test_network_spectra_lie_below_the_real_one_except_at_k_2():
    ratios = spectrum_ratios()
    for ratio in ratios:
        assert [k + 1 for k, r in enumerate(ratio) if r >= 1] == [2]
    middle = [r for ratio in ratios for r in ratio[2:6]]
    last = {f"{ratio[20]:.2f}" for ratio in ratios}
    assert len(last) == 1
    carries(
        passage("Both networks fall below it"),
        "at every shell except k = 2",
        f"to {min(middle):.2f} to {max(middle):.2f} of it for k = 3 to 6 "
        f"and {last.pop()} at k = 21",
    )
    carries(
        passage("Where the spectrum decides"),
        "lies below the real one at every shell except k = 2",
    )


def test_gaussian_and_network_relaxation_figures():
    c = results("consistency.json")["sources"]
    gauss = ("g_shell", "g_mode", "g_mode_ddim")
    nets = ("diffusion", "diffusion_plain")
    spread = [c[n]["band_log_ratio"]["all"] for n in gauss]
    lost = [-100 * c[n][key] for n in gauss for key in CHANGES]
    nets_all = [c[n]["band_log_ratio"]["all"] for n in nets]
    real = c["real_test"]["band_log_ratio"]["all"]
    head = (
        f"changes by {min(spread):.2f} to {max(spread):.2f}",
        f"{nets_all[0]:.2g} and {nets_all[1]:.2g} for the networks",
        f"{real:.1g} for real fields",
    )
    carries(passage("Where it does not, the diffusion model"), *head)
    carries(
        passage("The spectra of the Gaussian fields relax more"),
        *head,
        f"{min(lost):.0f} to {max(lost):.0f} percent of the energy and enstrophy",
    )
    first20 = [relaxation(c[n]["spectrum_ratio_after_over_before"], 20) for n in nets]
    gauss20 = [relaxation(c[n]["spectrum_ratio_after_over_before"], 20) for n in gauss]
    real20 = relaxation(c["real_test"]["spectrum_ratio_after_over_before"], 20)
    for name, ratio in zip(nets, first20, strict=True):
        assert ratio < min(gauss20), name
    gauss_range = f"{min(gauss20):.2f} to {max(gauss20):.2f}"
    carries(
        passage("Where it does not, the diffusion model"),
        f"over shells 1 to 20: {gauss_range} against {first20[0]:.3g} "
        f"and {first20[1]:.3g}",
    )
    carries(
        passage("Over shells 1 to 20 (column"),
        f"the scores are {first20[0]:.3g} and {first20[1]:.3g} for the networks, "
        f"{gauss_range} for the Gaussian fields and {real20:.1g} for real fields",
    )


def test_the_spectrum_at_the_last_shell_rises_for_both_networks():
    c = results("consistency.json")["sources"]
    rise = [
        100 * (c[n]["spectrum_ratio_after_over_before"][20] - 1)
        for n in ("diffusion", "diffusion_plain")
    ]
    carries(
        passage("For both networks the spectrum at k = 21"),
        f"rises by {rise[0]:.0f} and {rise[1]:.0f} percent",
    )


def test_the_reconstruction_gain_of_the_forcing_channel():
    plain = results("reconstruction_plain.json")["results"]

    def gain(name):
        rows = results(name)["results"]
        return [
            100
            * (
                1
                - rows[p]["diffusion"]["relative_l2"]["mean"]
                / plain[p]["diffusion"]["relative_l2"]["mean"]
            )
            for p in ("1", "2", "5", "10")
        ]

    seed0, seed1 = (
        gain("reconstruction.json"),
        gain("reconstruction_forcing_seed1.json"),
    )
    assert all(9.5 < g < 10.7 for g in seed0)
    carries(
        passage("Reconstructions have about"),
        f"about 10 percent lower L2 error ({min(seed0):.1f} to {max(seed0):.1f} "
        f"percent at the four fractions of observed points for seed 0, "
        f"{min(seed1):.1f} to {max(seed1):.1f} percent for seed 1)",
    )


def test_the_spin_up_windows_against_the_snapshot_mean():
    s = results("solver.json")["stationarity"]
    ratio = [w / s["energy_mean"] for w in s["spin_up_energy_by_20_time_units"]]
    below = [100 * (1 - r) for r in ratio[:2]]
    above = [100 * (r - 1) for r in ratio[4:]]
    carries(
        passage("The energy of the last two 20-unit windows"),
        f"about {above[0]:.0f} and {above[1]:.0f} percent above",
        f"about {below[0]:.1f} and {below[1]:.1f} percent below",
    )
    carries(
        passage("The 100 time units of spin-up were not shown"),
        f"about {above[0]:.0f} and {above[1]:.0f} percent more energy",
    )


def test_the_resolution_ratios():
    check = results("solver.json")["checks"]["resolution"]
    ratio = check["ratio_data_over_fine_by_shell"]
    assert all(1 < ratio[k - 1] < 1.2 for k in (5, 10, 14))
    carries(
        passage("The 64 x 64 grid is coarse"),
        "within 20 percent at k = 5, 10 and 14",
        f"{ratio[17]:.1f} times larger at k = 18",
        f"{ratio[20]:.1f} times larger at k = 21",
    )


def test_the_end_of_the_cosine_schedule():
    end = float(Schedule().alphas_bar[-1])
    blown = [
        results(name)["start"]["0.0"]["enstrophy_over_validation"]
        for name in ("sampler_start.json", "sampler_start_forcing.json")
    ]
    carries(
        passage("The start of the DDIM chain matters"),
        f"a signal level of {end * 1e9:.1f}e-9",
        f"{min(blown) / 1e4:.1f}e4 to {max(blown) / 1e4:.1f}e4 times that of the",
    )
    carries(
        passage("At the end of the cosine schedule"),
        f"{end * 1e9:.1f} \\cdot 10^{{-9}}",
        f"{1 / math.sqrt(end) / 1e4:.0f} \\cdot 10^{{4}}",
    )


def test_the_range_of_the_validation_error_over_the_starts():
    spread = []
    for name in ("sampler_start.json", "sampler_start_forcing.json"):
        error = [
            s["band_log_ratio_all"]
            for level, s in results(name)["start"].items()
            if level != "0.0"
        ]
        spread.append(max(error) - min(error))
    text = f"{min(spread):.2f} to {max(spread):.2f}"
    carries(passage("Where the spectrum decides"), f"{text} above the lowest")
    carries(passage("How 1e-3 became the setting"), f"lowers the error by {text}")
