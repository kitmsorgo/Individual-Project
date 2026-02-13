from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .utils import ensure_dir, save_json
from .pinn import LossWeights, compute_losses

CSV_COLUMNS = [
    "step",
    "total",
    "pde",
    "ic",
    "bc",
    "data",
    "grad_norm",
    "elapsed_s",
    "val_total",
    "val_pde",
    "val_ic",
    "val_bc",
    "val_data",
]


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total += param_norm.item() ** 2
    return total ** 0.5


def _save_checkpoint(path: Path, model: nn.Module) -> None:
    ensure_dir(path.parent)
    torch.save({"state_dict": model.state_dict()}, path)


def _init_loss_csv(csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def _write_loss_row(csv_path: Path, row: Dict[str, float | int]) -> None:
    ensure_dir(csv_path.parent)
    safe_row = {k: row.get(k, math.nan) for k in CSV_COLUMNS}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(safe_row)


def _case_ckpt_path(run_dir: Path, stem: str, case_id: str | None) -> Path:
    if case_id:
        return run_dir / f"{case_id}_{stem}.pt"
    return run_dir / f"{stem}.pt"


def compute_losses_eval(
    model: nn.Module,
    batch: Any,
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Keep autograd enabled: PINN residual terms may require derivatives at eval time.
    loss, logs = compute_losses(model, batch, weights)
    out_logs = {k: float(v) for k, v in logs.items()}
    return loss, out_logs


def train_adam(
    model: nn.Module,
    batch: Any,
    weights: LossWeights,
    lr: float = 1e-3,
    steps: int = 20000,
    print_every: int = 200,
    run_dir: str | Path = "checkpoints/run",
    case_id: str | None = None,
    val_batch: Any | None = None,
    keep_best_k: int = 0,
) -> Path:
    run_dir = ensure_dir(run_dir)
    loss_csv = run_dir / "loss.csv"
    meta_json = run_dir / "meta.json"
    best_path = _case_ckpt_path(run_dir, "best", case_id)
    last_path = _case_ckpt_path(run_dir, "last", case_id)

    save_json(
        {
            "optimizer": "adam",
            "lr": lr,
            "steps": steps,
            "weights": asdict(weights),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keep_best_k": keep_best_k,
            "has_val_batch": val_batch is not None,
        },
        meta_json,
    )

    _init_loss_csv(loss_csv)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")
    best_snapshots: list[Path] = []
    t0 = time.time()

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, train_logs = compute_losses_eval(model, batch, weights)
        loss.backward()
        grad_norm = _grad_norm(model)
        opt.step()

        elapsed_s = float(time.time() - t0)
        row: Dict[str, float | int] = {
            "step": step,
            "total": train_logs.get("total", math.nan),
            "pde": train_logs.get("pde", math.nan),
            "ic": train_logs.get("ic", math.nan),
            "bc": train_logs.get("bc", math.nan),
            "data": train_logs.get("data", math.nan),
            "grad_norm": float(grad_norm),
            "elapsed_s": elapsed_s,
        }

        if val_batch is not None:
            _, val_logs = compute_losses_eval(model, val_batch, weights)
            row.update(
                {
                    "val_total": val_logs.get("total", math.nan),
                    "val_pde": val_logs.get("pde", math.nan),
                    "val_ic": val_logs.get("ic", math.nan),
                    "val_bc": val_logs.get("bc", math.nan),
                    "val_data": val_logs.get("data", math.nan),
                }
            )

        _write_loss_row(loss_csv, row)

        total_loss = float(train_logs.get("total", math.nan))
        if total_loss < best:
            best = total_loss
            _save_checkpoint(best_path, model)
            if keep_best_k > 0:
                snapshot = _case_ckpt_path(run_dir, f"best_step{step}", case_id)
                _save_checkpoint(snapshot, model)
                best_snapshots.append(snapshot)
                if len(best_snapshots) > keep_best_k:
                    old_snapshot = best_snapshots.pop(0)
                    if old_snapshot.exists():
                        old_snapshot.unlink()

        if step % print_every == 0 or step == steps:
            _save_checkpoint(last_path, model)

        if step % print_every == 0 or step == 1:
            print(
                f"[Adam] step {step}/{steps} | total={row['total']:.4e} "
                f"(pde={row['pde']:.2e}, ic={row['ic']:.2e}, bc={row['bc']:.2e}, data={row['data']:.2e}) "
                f"| grad={row['grad_norm']:.2e} | {elapsed_s:.1f}s"
            )

    return best_path


def train_lbfgs(
    model: nn.Module,
    batch: Any,
    weights: LossWeights,
    max_iter: int = 2000,
    history_size: int = 50,
    lr: float = 1.0,
    run_dir: str | Path = "checkpoints/run",
    case_id: str | None = None,
    val_batch: Any | None = None,
) -> Path:
    """
    L-BFGS polish. Runs full-batch optimization. In PINNs, this often improves convergence.

    Note: PyTorch LBFGS calls closure multiple times per step.
    """
    run_dir = ensure_dir(run_dir)
    loss_csv = run_dir / "loss.csv"
    meta_json = run_dir / "meta.json"
    best_path = _case_ckpt_path(run_dir, "best", case_id)
    last_path = _case_ckpt_path(run_dir, "last", case_id)

    save_json(
        {
            "optimizer": "lbfgs",
            "lr": lr,
            "max_iter": max_iter,
            "history_size": history_size,
            "weights": asdict(weights),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "has_val_batch": val_batch is not None,
        },
        meta_json,
    )

    _init_loss_csv(loss_csv)
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss, _ = compute_losses(model, batch, weights)
        loss.backward()
        return loss

    print("[LBFGS] starting...")
    t0 = time.time()
    opt.step(closure)

    # Final evaluation is done with autograd available: PINN residual terms may require derivatives.
    model.zero_grad(set_to_none=True)
    total, train_logs = compute_losses_eval(model, batch, weights)
    total.backward()
    grad_norm = _grad_norm(model)

    elapsed_s = float(time.time() - t0)
    row: Dict[str, float | int] = {
        "step": max_iter,
        "total": train_logs.get("total", math.nan),
        "pde": train_logs.get("pde", math.nan),
        "ic": train_logs.get("ic", math.nan),
        "bc": train_logs.get("bc", math.nan),
        "data": train_logs.get("data", math.nan),
        "grad_norm": float(grad_norm),
        "elapsed_s": elapsed_s,
    }

    if val_batch is not None:
        _, val_logs = compute_losses_eval(model, val_batch, weights)
        row.update(
            {
                "val_total": val_logs.get("total", math.nan),
                "val_pde": val_logs.get("pde", math.nan),
                "val_ic": val_logs.get("ic", math.nan),
                "val_bc": val_logs.get("bc", math.nan),
                "val_data": val_logs.get("data", math.nan),
            }
        )

    _write_loss_row(loss_csv, row)
    _save_checkpoint(best_path, model)
    _save_checkpoint(last_path, model)

    print(
        f"[LBFGS] done | total={row['total']:.4e} "
        f"(pde={row['pde']:.2e}, ic={row['ic']:.2e}, bc={row['bc']:.2e}, data={row['data']:.2e}) "
        f"| grad={row['grad_norm']:.2e}"
    )
    return best_path
