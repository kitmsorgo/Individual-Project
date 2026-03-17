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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Tuple, Callable

import torch
import torch.nn as nn

from .utils import ensure_dir, save_json
from .pinn import LossWeights, FluxLearnedConfig, compute_losses

CSV_COLUMNS = [
    "step",
    "total",
    "pde",
    "ic",
    "bc",
    "bc_left",
    "bc_right",
    "bc_left_rmse",
    "bc_right_rmse",
    "bc_monitor",
    "data",
    "grad_loss",
    "rmse_data",
    "grad_norm",
    "elapsed_s",
    "val_total",
    "val_pde",
    "val_ic",
    "val_bc",
    "val_bc_left",
    "val_bc_right",
    "val_bc_left_rmse",
    "val_bc_right_rmse",
    "val_data",
    "val_grad_loss",
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
    create_graph: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluate losses while keeping autograd enabled.

    For PINNs, some residual terms require computing derivatives at evaluation
    time, so we return the loss tensor (with grad history) and a dict of
    scalar log values.

    The `create_graph` flag can be set to False during validation to avoid
    retaining computational graphs when gradients are not needed.
    """
    loss, logs = compute_losses(
        model,
        batch,
        weights, 
        create_graph=create_graph,
    )
    out_logs = {k: float(v) for k, v in logs.items()}
    return loss, out_logs

def _collect_trainable_params(
    model: nn.Module,
    extra_params: list[nn.Parameter] | None,
) -> list[nn.Parameter]:
    params = list(model.parameters())

    if extra_params:
        params.extend(list(extra_params))

    # de-duplicate while preserving order
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(p)
    return unique


def train_adam(
    model: nn.Module,
    batch: Any,
    weights: LossWeights, 
    extra_params: list[nn.Parameter] | None = None,
    log_callback: Callable[[Dict[str, float | int | str | None]], None] | None = None,
    log_every: int = 1,
    lr: float = 1e-3,
    steps: int | None = None,
    print_every: int = 200,
    run_dir: str | Path = "models/checkpoints/run",
    case_id: str | None = None,
    val_batch: Any | None = None,
    keep_best_k: int = 0,
    eval_every: int = 500,
    max_steps: int = 20000,
    min_steps: int = 200,
    patience_evals: int = 8,
    plateau_window: int = 5,
    plateau_rel_tol: float = 0.01,
    pde_guardrail_rel: float = 0.10,
    data_batch_size: int | None = None,
) -> Path:
    """Train `model` with Adam using validation-RMSE-based convergence control.

    Key steps per iteration:
    - zero gradients
    - compute losses (keeps autograd for PINN derivatives)
    - backward() to populate gradients
    - compute gradient norm for logging
    - optimizer step
    - periodic validation checkpoints, convergence checks, and CSV logging

    Checkpoint selection and early stopping are driven by validation RMSE.
    Validation PDE is still logged for monitoring, but it does not block a
    checkpoint that improves RMSE.

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
            "data_batch_size": data_batch_size,
            "weights": asdict(weights),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keep_best_k": keep_best_k,
            "has_val_batch": val_batch is not None,
        },
        meta_json,
    )

    _init_loss_csv(loss_csv)
    params = _collect_trainable_params(model, extra_params)
    opt = torch.optim.Adam(params, lr=lr)

    # Backward compatibility: if `steps` is passed, treat it as max_steps.
    if steps is not None:
        max_steps = int(steps)

    # Without validation we cannot make a robust convergence decision, so fall back to max_steps.
    use_val_controller = val_batch is not None

    best = float("inf")
    best_snapshots: list[Path] = []
    best_val_rmse = float("inf")
    no_improve_evals = 0
    rmse_eval_history: list[float] = []
    stop_reason = "reached_max_steps"
    t0 = time.time()

    for step in range(1, max_steps + 1):
        # Standard Adam iteration
        opt.zero_grad(set_to_none=True)

        # Subsample supervised data for runtime savings (training only)
        batch_for_loss = batch
        if (
            data_batch_size is not None
            and getattr(batch, "xi_data", None) is not None
            and batch.xi_data.numel() > 0
        ):
            n_data = batch.xi_data.shape[0]
            if n_data > data_batch_size:
                idx = torch.randperm(n_data, device=batch.xi_data.device)[:data_batch_size]
                batch_for_loss = replace(
                    batch,
                    xi_data=batch.xi_data[idx],
                    tau_data=batch.tau_data[idx],
                    mu_data=batch.mu_data[idx] if getattr(batch, "mu_data", None) is not None else None,
                    theta_data=batch.theta_data[idx],
                )

        loss, train_logs = compute_losses_eval(
            model,
            batch_for_loss,
            weights, 
            create_graph=True,
        )
        loss.backward()
        del loss
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
            "bc_left": train_logs.get("bc_left", math.nan),
            "bc_right": train_logs.get("bc_right", math.nan),
            "bc_left_rmse": train_logs.get("bc_left_rmse", math.nan),
            "bc_right_rmse": train_logs.get("bc_right_rmse", math.nan),
            "bc_monitor": train_logs.get("bc_monitor", math.nan),
            "data": train_logs.get("data", math.nan),
            "grad_loss": train_logs.get("grad_loss", math.nan),
            "rmse_data": train_data_rmse,
            "grad_norm": float(grad_norm),
            "elapsed_s": elapsed_s,
        }

        eval_now = use_val_controller and (step % eval_every == 0 or step == 1)
        if eval_now:
            _, val_logs = compute_losses_eval(
                model,
                val_batch,
                weights, 
                create_graph=False,
            )
            val_data_mse = float(val_logs.get("data", math.nan))
            val_rmse_data = _rmse_from_mse(val_data_mse)
            gen_gap = val_rmse_data - train_data_rmse
            row.update(
                {
                    "val_total": val_logs.get("total", math.nan),
                    "val_pde": val_logs.get("pde", math.nan),
                    "val_ic": val_logs.get("ic", math.nan),
                    "val_bc": val_logs.get("bc", math.nan),
                    "val_bc_left": val_logs.get("bc_left", math.nan),
                    "val_bc_right": val_logs.get("bc_right", math.nan),
                    "val_bc_left_rmse": val_logs.get("bc_left_rmse", math.nan),
                    "val_bc_right_rmse": val_logs.get("bc_right_rmse", math.nan),
                    "val_data": val_data_mse,
                    "val_grad_loss": val_logs.get("grad_loss", math.nan),
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
                    "val_bc_left": math.nan,
                    "val_bc_right": math.nan,
                    "val_bc_left_rmse": math.nan,
                    "val_bc_right_rmse": math.nan,
                    "val_data": math.nan,
                    "val_grad_loss": math.nan,
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
            rmse_eval_history.append(current_rmse)
            
            is_improvement = False
            if math.isfinite(current_rmse):
                if not math.isfinite(best_val_rmse):
                    is_improvement = True
                    
                else:
                    rmse_ok, _ = _is_meaningful_rmse_improvement(current_rmse, best_val_rmse)
                    is_improvement = rmse_ok

            if is_improvement:
                print(f"There has been an improvement in validation RMSE data. {best_val_rmse} -> {current_rmse}")
                best_val_rmse = current_rmse
                no_improve_evals = 0
                _save_checkpoint(best_path, model)
            else:
                print("No meaningful improvement in validation RMSE data.")
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

        if log_callback is not None and (step % max(1, int(log_every)) == 0):
            try:
                log_callback(
                    {
                        "step": step,
                        "stage": "adam",
                        "total_loss": float(row.get("total", math.nan)),
                        "pde_loss": float(row.get("pde", math.nan)),
                        "bc_loss": float(row.get("bc", math.nan)),
                        "bc_left_loss": float(row.get("bc_left", math.nan)),
                        "bc_right_loss": float(row.get("bc_right", math.nan)),
                        "bc_left_rmse": float(row.get("bc_left_rmse", math.nan)),
                        "bc_right_rmse": float(row.get("bc_right_rmse", math.nan)),
                        "ic_loss": float(row.get("ic", math.nan)),
                        "data_loss": float(row.get("data", math.nan)),
                        "grad_loss": float(row.get("grad_loss", math.nan)),
                        "grad_norm": float(row.get("grad_norm", math.nan)),
                        "learning_rate": float(lr),
                        "time_elapsed_s": float(elapsed_s),
                    }
                )
            except Exception as exc:
                print(f"[train_adam] log_callback failed: {exc}")

        if step % print_every == 0 or step == max_steps:
            _save_checkpoint(last_path, model)

        if step % print_every == 0 or step == 1:
            print(
                f"[Adam] step {step}/{max_steps} | total={row['total']:.4e} "
                f"(pde={row['pde']:.2e}, ic={row['ic']:.2e}, bcL={row['bc_left']:.2e}, "
                f"bcR={row['bc_right']:.2e}, data={row['data']:.2e}, grad={row['grad_loss']:.2e}) "
                f"| bc_rmse(L={row['bc_left_rmse']:.2e}, R={row['bc_right_rmse']:.2e}) "
                f"| val_rmse={row['val_rmse_data']:.2e} | val_bc_rmse(L={row['val_bc_left_rmse']:.2e}, "
                f"R={row['val_bc_right_rmse']:.2e}) | val_pde={row['val_pde']:.2e} "
                f"| val_grad={row['val_grad_loss']:.2e} "
                f"| grad={row['grad_norm']:.2e} | {elapsed_s:.1f}s"
            )

    print(f"[Adam] done | stop_reason={stop_reason}")
    return best_path


def train_lbfgs(
    model: nn.Module,
    batch: Any,
    weights: LossWeights, 
    extra_params: list[nn.Parameter] | None = None,
    log_callback: Callable[[Dict[str, float | int | str | None]], None] | None = None,
    log_every: int = 1,
    print_every: int = 200,
    max_iter: int = 2000,
    history_size: int = 50,
    lr: float = 1.0,
    run_dir: str | Path = "models/checkpoints/run",
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
            "print_every": print_every,
            "weights": asdict(weights),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "has_val_batch": val_batch is not None,
        },
        meta_json,
    )

    _init_loss_csv(loss_csv)
    params = _collect_trainable_params(model,  extra_params)
    opt = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        opt.zero_grad(set_to_none=True)
        loss, _ = compute_losses(model, batch, weights )
        loss.backward()
        closure_calls += 1
        if log_callback is not None and (closure_calls % max(1, int(log_every)) == 0):
            try:
                grad_norm = _grad_norm(model)
                log_callback(
                    {
                        "step": closure_calls,
                        "stage": "lbfgs",
                        "total_loss": float(loss.detach().cpu().item()),
                        "pde_loss": math.nan,
                        "bc_loss": math.nan,
                        "ic_loss": math.nan,
                        "data_loss": math.nan,
                        "grad_norm": float(grad_norm),
                        "learning_rate": "",
                        "time_elapsed_s": float(time.time() - t0),
                    }
                )
            except Exception as exc:
                print(f"[train_lbfgs] log_callback failed: {exc}")
        if closure_calls % max(1, int(print_every)) == 0:
            elapsed_s = float(time.time() - t0)
            print(
                f"[LBFGS] closure {closure_calls} | "
                f"loss={float(loss.detach().cpu().item()):.4e} | "
                f"{elapsed_s:.1f}s"
            )
        return loss

    print("[LBFGS] starting...")
    t0 = time.time()
    opt.step(closure)

    # Final evaluation is done with autograd available: PINN residual terms may require derivatives.
    # Evaluate final loss and gradient norm for logging
    model.zero_grad(set_to_none=True)
    total, train_logs = compute_losses_eval(
        model, batch, weights 
    )
    total.backward()
    grad_norm = _grad_norm(model)

    elapsed_s = float(time.time() - t0)
    row: Dict[str, float | int] = {
        "step": max_iter,
        "total": train_logs.get("total", math.nan),
        "pde": train_logs.get("pde", math.nan),
        "ic": train_logs.get("ic", math.nan),
        "bc": train_logs.get("bc", math.nan),
        "bc_left": train_logs.get("bc_left", math.nan),
        "bc_right": train_logs.get("bc_right", math.nan),
        "bc_left_rmse": train_logs.get("bc_left_rmse", math.nan),
        "bc_right_rmse": train_logs.get("bc_right_rmse", math.nan),
        "data": train_logs.get("data", math.nan),
        "grad_loss": train_logs.get("grad_loss", math.nan),
        "grad_norm": float(grad_norm),
        "elapsed_s": elapsed_s,
    }

    if val_batch is not None:
        _, val_logs = compute_losses_eval(
            model, val_batch, weights  
        )
        row.update(
            {
                "val_total": val_logs.get("total", math.nan),
                "val_pde": val_logs.get("pde", math.nan),
                "val_ic": val_logs.get("ic", math.nan),
                "val_bc": val_logs.get("bc", math.nan),
                "val_bc_left": val_logs.get("bc_left", math.nan),
                "val_bc_right": val_logs.get("bc_right", math.nan),
                "val_bc_left_rmse": val_logs.get("bc_left_rmse", math.nan),
                "val_bc_right_rmse": val_logs.get("bc_right_rmse", math.nan),
                "val_data": val_logs.get("data", math.nan),
                "val_grad_loss": val_logs.get("grad_loss", math.nan),
            }
        )

    _write_loss_row(loss_csv, row)
    _save_checkpoint(best_path, model)
    _save_checkpoint(last_path, model)

    print(
        f"[LBFGS] done | total={row['total']:.4e} "
        f"(pde={row['pde']:.2e}, ic={row['ic']:.2e}, bcL={row['bc_left']:.2e}, "
        f"bcR={row['bc_right']:.2e}, data={row['data']:.2e}, grad={row['grad_loss']:.2e}) "
        f"| bc_rmse(L={row['bc_left_rmse']:.2e}, R={row['bc_right_rmse']:.2e}) "
        f"| grad={row['grad_norm']:.2e}"
    )
    return best_path
