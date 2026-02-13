from __future__ import annotations

"""Training helpers for PINN experiments.

This module contains convenience functions used by the training scripts:
- small helpers for checkpointing and logging
- implementations of `train_adam` and `train_lbfgs` routines

Both optimizers expect a full-batch `batch` object and a `LossWeights`
instance from `PINN/src/pinn.py`.
"""

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
    "rmse_data",
    "grad_norm",
    "elapsed_s",
    "val_total",
    "val_pde",
    "val_ic",
    "val_bc",
    "val_data",
    "val_rmse_data",
    "gen_gap_rmse_data",
]


def _grad_norm(model: nn.Module) -> float:
    """Compute global L2 norm of gradients across model parameters.

    Returns a scalar float used for logging / monitoring training stability.
    """
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


def _rmse_from_mse(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        return math.nan
    return math.sqrt(value)


def _is_meaningful_rmse_improvement(current_rmse: float, best_rmse: float) -> tuple[bool, float]:
    """Check if RMSE gain is meaningful using max(1% relative, 1e-4 absolute)."""
    if not (math.isfinite(current_rmse) and math.isfinite(best_rmse)):
        return False, math.nan
    gain = best_rmse - current_rmse
    threshold = max(0.01 * best_rmse, 1e-4)
    return gain >= threshold, threshold


def compute_losses_eval(
    model: nn.Module,
    batch: Any,
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluate losses while keeping autograd enabled.

    For PINNs, some residual terms require computing derivatives at evaluation
    time, so we return the loss tensor (with grad history) and a dict of
    scalar log values.
    """
    # Keep autograd enabled: PINN residual terms may require derivatives at eval time.
    loss, logs = compute_losses(model, batch, weights)
    out_logs = {k: float(v) for k, v in logs.items()}
    return loss, out_logs


def train_adam(
    model: nn.Module,
    batch: Any,
    weights: LossWeights,
    lr: float = 1e-3,
    steps: int | None = None,
    print_every: int = 200,
    run_dir: str | Path = "checkpoints/run",
    case_id: str | None = None,
    val_batch: Any | None = None,
    keep_best_k: int = 0,
    eval_every: int = 100,
    max_steps: int = 20000,
    min_steps: int = 200,
    patience_evals: int = 8,
    plateau_window: int = 5,
    plateau_rel_tol: float = 0.01,
    pde_guardrail_rel: float = 0.10,
) -> Path:
    """Train `model` with Adam using validation-based convergence control.

    Key steps per iteration:
    - zero gradients
    - compute losses (keeps autograd for PINN derivatives)
    - backward() to populate gradients
    - compute gradient norm for logging
    - optimizer step
    - periodic validation checkpoints, convergence checks, and CSV logging

    Why this controller avoids fixed step counts:
    - A fixed step count such as 1000 is arbitrary and may underfit or overfit.
    - Validation metrics should drive stopping: they often improve early and then plateau.
    - Relative change over recent checkpoints (sliding window) is a practical convergence signal.
    - Optimal training length depends on architecture, optimizer dynamics, and data.

    Returns path to the best checkpoint saved.
    """
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
            "eval_every": eval_every,
            "max_steps": max_steps,
            "min_steps": min_steps,
            "patience_evals": patience_evals,
            "plateau_window": plateau_window,
            "plateau_rel_tol": plateau_rel_tol,
            "pde_guardrail_rel": pde_guardrail_rel,
            "weights": asdict(weights),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keep_best_k": keep_best_k,
            "has_val_batch": val_batch is not None,
        },
        meta_json,
    )

    _init_loss_csv(loss_csv)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Backward compatibility: if `steps` is passed, treat it as max_steps.
    if steps is not None:
        max_steps = int(steps)

    # Without validation we cannot make a robust convergence decision, so fall back to max_steps.
    use_val_controller = val_batch is not None

    best = float("inf")
    best_snapshots: list[Path] = []
    best_val_rmse = float("inf")
    best_val_pde = float("inf")
    no_improve_evals = 0
    rmse_eval_history: list[float] = []
    stop_reason = "reached_max_steps"
    t0 = time.time()

    for step in range(1, max_steps + 1):
        # Standard Adam iteration
        opt.zero_grad(set_to_none=True)
        loss, train_logs = compute_losses_eval(model, batch, weights)
        loss.backward()
        grad_norm = _grad_norm(model)
        opt.step()

        elapsed_s = float(time.time() - t0)
        train_data_rmse = _rmse_from_mse(float(train_logs.get("data", math.nan)))
        row: Dict[str, float | int] = {
            "step": step,
            "total": train_logs.get("total", math.nan),
            "pde": train_logs.get("pde", math.nan),
            "ic": train_logs.get("ic", math.nan),
            "bc": train_logs.get("bc", math.nan),
            "data": train_logs.get("data", math.nan),
            "rmse_data": train_data_rmse,
            "grad_norm": float(grad_norm),
            "elapsed_s": elapsed_s,
        }

        eval_now = use_val_controller and (step % eval_every == 0 or step == 1)
        if eval_now:
            _, val_logs = compute_losses_eval(model, val_batch, weights)
            val_data_mse = float(val_logs.get("data", math.nan))
            val_rmse_data = _rmse_from_mse(val_data_mse)
            gen_gap = val_rmse_data - train_data_rmse
            row.update(
                {
                    "val_total": val_logs.get("total", math.nan),
                    "val_pde": val_logs.get("pde", math.nan),
                    "val_ic": val_logs.get("ic", math.nan),
                    "val_bc": val_logs.get("bc", math.nan),
                    "val_data": val_data_mse,
                    "val_rmse_data": val_rmse_data,
                    "gen_gap_rmse_data": gen_gap,
                }
            )
        else:
            row.update(
                {
                    "val_total": math.nan,
                    "val_pde": math.nan,
                    "val_ic": math.nan,
                    "val_bc": math.nan,
                    "val_data": math.nan,
                    "val_rmse_data": math.nan,
                    "gen_gap_rmse_data": math.nan,
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

        if eval_now:
            current_rmse = float(row.get("val_rmse_data", math.nan))
            current_val_pde = float(row.get("val_pde", math.nan))
            rmse_eval_history.append(current_rmse)

            is_improvement = False
            if math.isfinite(current_rmse):
                if not math.isfinite(best_val_rmse):
                    is_improvement = True
                else:
                    rmse_ok, _ = _is_meaningful_rmse_improvement(current_rmse, best_val_rmse)
                    pde_guard_ok = True
                    if math.isfinite(best_val_pde) and math.isfinite(current_val_pde):
                        pde_guard_ok = current_val_pde <= best_val_pde * (1.0 + pde_guardrail_rel)
                    elif math.isfinite(best_val_pde) and not math.isfinite(current_val_pde):
                        pde_guard_ok = False
                    is_improvement = rmse_ok and pde_guard_ok

            if is_improvement:
                best_val_rmse = current_rmse
                best_val_pde = current_val_pde
                no_improve_evals = 0
                _save_checkpoint(best_path, model)
            else:
                no_improve_evals += 1

            plateau = False
            if len(rmse_eval_history) >= plateau_window + 1:
                old_rmse = rmse_eval_history[-(plateau_window + 1)]
                new_rmse = rmse_eval_history[-1]
                if math.isfinite(old_rmse) and math.isfinite(new_rmse):
                    rel_gain = (old_rmse - new_rmse) / max(abs(old_rmse), 1e-12)
                    plateau = rel_gain < plateau_rel_tol

            if step >= min_steps and (no_improve_evals >= patience_evals or plateau):
                stop_reason = (
                    f"early_stop_patience({no_improve_evals}/{patience_evals})"
                    if no_improve_evals >= patience_evals
                    else f"early_stop_plateau(rel_gain<{plateau_rel_tol:.3f} over {plateau_window} evals)"
                )
                _save_checkpoint(last_path, model)
                print(f"[Adam] stopping at step {step}: {stop_reason}")
                break

        if step % print_every == 0 or step == max_steps:
            _save_checkpoint(last_path, model)

        if step % print_every == 0 or step == 1:
            print(
                f"[Adam] step {step}/{max_steps} | total={row['total']:.4e} "
                f"(pde={row['pde']:.2e}, ic={row['ic']:.2e}, bc={row['bc']:.2e}, data={row['data']:.2e}) "
                f"| val_rmse={row['val_rmse_data']:.2e} | val_pde={row['val_pde']:.2e} "
                f"| grad={row['grad_norm']:.2e} | {elapsed_s:.1f}s"
            )

    print(f"[Adam] done | stop_reason={stop_reason}")
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
    # Evaluate final loss and gradient norm for logging
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
