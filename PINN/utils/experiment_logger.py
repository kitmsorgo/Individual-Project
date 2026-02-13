from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

CSV_COLUMNS = [
    "timestamp",
    "git_commit",
    "seed",
    "n_layers",
    "n_neurons",
    "activation",
    "optimizer",
    "epochs",
    "lr",
    "n_colloc",
    "n_boundary",
    "n_initial",
    "n_data",
    "loss_total",
    "loss_pde",
    "loss_ic",
    "loss_bc",
    "loss_data",
    "val_rmse_data",
    "val_pde_loss",
]


def _to_float(value: object) -> float:
    try:
        if value is None:
            return math.nan
        if isinstance(value, str) and value.strip() == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def ensure_csv(path: str) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_result(path: str, result: dict) -> None:
    ensure_csv(path)
    csv_path = Path(path)
    row = {col: result.get(col, "") for col in CSV_COLUMNS}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def load_best(path: str, metric: str = "val_rmse_data") -> dict | None:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            return None

        best_row: dict | None = None
        best_metric = math.inf
        for row in reader:
            metric_val = _to_float(row.get(metric))
            if not math.isfinite(metric_val):
                continue
            if metric_val < best_metric:
                best_metric = metric_val
                best_row = row
        return best_row


def _fmt(value: object) -> str:
    fval = _to_float(value)
    if math.isfinite(fval):
        return f"{fval:.6e}"
    return "nan"


def compare_to_best(current: dict, best: dict | None) -> Tuple[bool, str]:
    if best is None:
        return True, "No baseline row exists yet; first run is baseline best."

    cur_rmse = _to_float(current.get("val_rmse_data"))
    best_rmse = _to_float(best.get("val_rmse_data"))
    if not math.isfinite(cur_rmse):
        return False, "Current val_rmse_data is missing/NaN."
    if not math.isfinite(best_rmse):
        return True, "Baseline val_rmse_data is missing/NaN; accepting current run."

    min_required = max(0.01 * best_rmse, 1e-4)
    improvement = best_rmse - cur_rmse
    if improvement < min_required:
        return (
            False,
            f"RMSE gain {improvement:.6e} is below threshold {min_required:.6e}.",
        )

    cur_pde = _to_float(current.get("val_pde_loss"))
    best_pde = _to_float(best.get("val_pde_loss"))
    if math.isfinite(cur_pde) and math.isfinite(best_pde):
        max_allowed = 1.10 * best_pde
        if cur_pde > max_allowed:
            rel = (cur_pde / best_pde - 1.0) * 100.0 if best_pde != 0 else math.inf
            return (
                False,
                f"PDE guardrail failed: {cur_pde:.6e} vs baseline {best_pde:.6e} ({rel:.2f}% worse).",
            )
    elif math.isfinite(best_pde) and not math.isfinite(cur_pde):
        return False, "PDE guardrail failed: current val_pde_loss is missing/NaN."

    return (
        True,
        f"RMSE improved by {improvement:.6e} (baseline {best_rmse:.6e} -> current {cur_rmse:.6e}) with PDE guardrail satisfied.",
    )

