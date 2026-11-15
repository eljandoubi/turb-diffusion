"""Train the periodic UNet DDPM on the vorticity fields.

Writes results/train.json and results/train_log.csv (train_x.json and
train_log_x.csv with --tag _x). The weights go to --work and are not tracked:
one EMA checkpoint per evaluation, and ema.pt, the one with the lowest held-out
noise-prediction error. A GPU is recommended; on a CPU use a few steps to test.
"""

import argparse
import csv
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from turbdiff.data import load, merge_json, random_symmetries
from turbdiff.diffusion import EMA, Schedule, noisy, training_loss
from turbdiff.solver import Params
from turbdiff.unet import UNet, count_parameters

ROOT = Path(__file__).resolve().parents[1]


def learning_rate(step: int, steps: int, lr: float, warmup: int) -> float:
    decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))
    return lr * min(1.0, (step + 1) / warmup) * decay


@torch.no_grad()
def validation_loss(model, sched, fields, device) -> float:
    """Noise-prediction error on held-out fields, with fixed timesteps and noise."""
    fields = torch.cat([fields, fields])
    g = torch.Generator().manual_seed(12345)
    noise = torch.randn(fields.shape, generator=g).to(device)
    t = ((torch.arange(len(fields)) * 397) % sched.steps).to(device)
    pred = model(noisy(sched, fields, t, noise), t)
    return float(((pred.float() - noise) ** 2).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "turb.npz")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--forcing", action="store_true", help="forcing input channel")
    parser.add_argument("--tag", default="", help="suffix of the result file names")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    data = load(args.data)
    n = data["train"].shape[-1]
    train = (data["train"].reshape(-1, 1, n, n) / data["sigma"]).to(device)
    val = (data["val"].reshape(-1, 1, n, n) / data["sigma"]).to(device)
    period_y = n // Params().kf

    model = UNet(
        base=args.base,
        dropout=args.dropout,
        forcing_wavenumber=Params().kf if args.forcing else 0,
    ).to(device)
    ema = EMA(model, args.ema)
    net = torch.compile(model) if args.compile else model
    sched = Schedule()
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=amp)
    g = torch.Generator(device=device).manual_seed(args.seed)

    args.work.mkdir(parents=True, exist_ok=True)
    log, running, start = [], [], time.time()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup)
        pick = torch.randint(0, len(train), (args.batch,), generator=g, device=device)
        batch = random_symmetries(train[pick], period_y, g)
        with torch.autocast(device, dtype=torch.float16) if amp else nullcontext():
            loss = training_loss(net, sched, batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model, step)
        running.append(loss.item())
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            row = {
                "step": step + 1,
                "train_loss": sum(running) / len(running),
                "val_loss_ema": validation_loss(ema.model, sched, val, device),
                "seconds": time.time() - start,
            }
            running = []
            log.append(row)
            print(row, flush=True)
            torch.save(ema.model.state_dict(), args.work / f"ema_{step + 1}.pt")

    seconds = time.time() - start
    best = min(log, key=lambda row: row["val_loss_ema"])
    shutil.copy(args.work / f"ema_{best['step']}.pt", args.work / "ema.pt")
    args.results.mkdir(parents=True, exist_ok=True)
    with open(args.results / f"train_log{args.tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    merge_json(
        args.results / f"train{args.tag}.json",
        {
            "device": torch.cuda.get_device_name(0) if amp else "cpu",
            "fp16_autocast": amp,
            "parameters": count_parameters(model),
            "base_channels": args.base,
            "forcing_channel": args.forcing,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "torch_compile": args.compile,
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "warmup": args.warmup,
            "ema_decay": args.ema,
            "training_fields": len(train),
            "wall_clock_seconds": seconds,
            "steps_per_second": args.steps / seconds,
            "final_train_loss": log[-1]["train_loss"],
            "final_val_loss_ema": log[-1]["val_loss_ema"],
            "selected_step": best["step"],
            "selected_val_loss_ema": best["val_loss_ema"],
            "seed": args.seed,
        },
    )
    out = args.results / f"train{args.tag}.json"
    print(f"selected step {best['step']}, wrote {out}")


if __name__ == "__main__":
    main()
