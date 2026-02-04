from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .utils import ensure_dir, save_json
from .pinn import LossWeights, compute_losses


def _save_checkpoint(path: Path, model: nn.Module) -> None:
    ensure_dir(path.parent)
    torch.save({"state_dict": model.state_dict()}, path)


def _write_loss_row(csv_path: Path, row: Dict[str, float], write_header: bool) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_adam(
    model: nn.Module,
    batch,
    weights: LossWeights,
    lr: float = 1e-3,
    steps: int = 20000,
    print_every: int = 200,
    run_dir: str | Path = "models/checkpoints/run",
) -> Path:
    run_dir = ensure_dir(run_dir)
    loss_csv = run_dir / "loss.csv"
    meta_json = run_dir / "meta.json"
    best_path = run_dir / "best.pt"

    save_json(
        {
            "optimizer": "adam",
            "lr": lr,
            "steps": steps,
            "weights": asdict(weights),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        meta_json,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")
    wrote_header = False
    t0 = time.time()

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, logs = compute_losses(model, batch, weights)
        loss.backward()
        opt.step()

        row = {"step": step, **logs}
        _write_loss_row(loss_csv, row, write_header=not wrote_header)
        wrote_header = True

        if logs["total"] < best:
            best = logs["total"]
            _save_checkpoint(best_path, model)

        if step % print_every == 0 or step == 1:
            dt = time.time() - t0
            print(
                f"[Adam] step {step}/{steps} | total={logs['total']:.4e} "
                f"(pde={logs['pde']:.2e}, ic={logs['ic']:.2e}, bc={logs['bc']:.2e}, data={logs['data']:.2e}) "
                f"| {dt:.1f}s"
            )

    return best_path


def train_lbfgs(
    model: nn.Module,
    batch,
    weights: LossWeights,
    max_iter: int = 2000,
    history_size: int = 50,
    lr: float = 1.0,
    run_dir: str | Path = "models/checkpoints/run",
) -> Path:
    """
    L-BFGS polish. Runs full-batch optimization. In PINNs, this often improves convergence.

    Note: PyTorch LBFGS calls closure multiple times per step.
    """
    run_dir = ensure_dir(run_dir)
    best_path = run_dir / "best_lbfgs.pt"

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    best = float("inf")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss, _ = compute_losses(model, batch, weights)
        loss.backward()
        return loss

    print("[LBFGS] starting...")
    loss = opt.step(closure)

    # One final evaluation for reporting
    with torch.no_grad():
        total, logs = compute_losses(model, batch, weights)
        if logs["total"] < best:
            best = logs["total"]
            _save_checkpoint(best_path, model)

    print(
        f"[LBFGS] done | total={logs['total']:.4e} "
        f"(pde={logs['pde']:.2e}, ic={logs['ic']:.2e}, bc={logs['bc']:.2e}, data={logs['data']:.2e})"
    )
    return best_path
