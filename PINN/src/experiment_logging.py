from __future__ import annotations

import csv
import json
import math
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
import subprocess

import torch


class ExperimentLogger:
    def __init__(self, results_dir: Path | str, flush_every: int = 10) -> None:
        self.results_dir = Path(results_dir)
        self.curves_dir = self.results_dir / "training_curves"
        self.config_dir = self.results_dir / "configs"
        self.master_path = self.results_dir / "experiment_log.csv"
        self.curve_path: Path | None = None
        self.config_path: Path | None = None
        self.flush_every = int(flush_every)
        self._buffer: list[Dict[str, object]] = []
        self._start_time: float | None = None
        self._config: Dict[str, object] = {}
        self.experiment_id: str | None = None
        self.master_fields = [
            "experiment_id",
            "date_time_utc",
            "commit_hash",
            "python_version",
            "torch_version",
            "cuda_available",
            "device",
            "random_seed",
            "case_id",
            "data_paths",
            "normalisation_type",
            "notes",
            "num_hidden_layers",
            "neurons_per_layer",
            "activation",
            "weight_init",
            "input_dim",
            "output_dim",
            "num_trainable_params",
            "lambda_total",
            "lambda_pde",
            "lambda_bc",
            "lambda_ic",
            "lambda_data",
            "n_collocation",
            "n_bc",
            "n_ic",
            "n_data",
            "collocation_sampler",
            "domain_bounds",
            "optimiser_stage1",
            "adam_lr",
            "adam_epochs",
            "optimiser_stage2",
            "lbfgs_max_iter",
            "lbfgs_history_size",
            "lbfgs_line_search_fn",
            "batch_size",
            "wall_clock_time_s",
            "convergence_flag",
            "final_total_loss",
            "final_pde_loss",
            "final_bc_loss",
            "final_ic_loss",
            "final_data_loss",
            "final_grad_norm",
            "l2_error_temperature",
            "max_abs_error",
            "relative_l2_error",
            "r2_score",
        ]
        self.curve_fields = [
            "step",
            "stage",
            "total_loss",
            "pde_loss",
            "bc_loss",
            "ic_loss",
            "data_loss",
            "grad_norm",
            "learning_rate",
            "time_elapsed_s",
        ]

    def _ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _write_header_if_needed(self, path: Path, fields: list[str]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

    def start_run(self, config: dict) -> str:
        try:
            self._ensure_dirs()
            self._start_time = time.time()
            short_hash = config.get("commit_hash", "unknown")
            rand = uuid.uuid4().hex[:6]
            date_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            case_id = str(config.get("case_id", "case"))
            self.experiment_id = f"{date_str}_{case_id}_{short_hash}_{rand}"
            self.curve_path = self.curves_dir / f"{self.experiment_id}.csv"
            self.config_path = self.config_dir / f"{self.experiment_id}.json"
            self._config = dict(config)

            self._write_header_if_needed(self.master_path, self.master_fields)
            self._write_header_if_needed(self.curve_path, self.curve_fields)

            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(self._config, f, indent=2)
            except Exception as exc:
                print(f"[ExperimentLogger] config dump failed: {exc}")

            return self.experiment_id
        except Exception as exc:
            print(f"[ExperimentLogger] start_run failed: {exc}")
            return "unknown"

    def log_step(self, metrics: dict) -> None:
        try:
            if self.curve_path is None:
                return
            row = {k: "" for k in self.curve_fields}
            row.update({k: metrics.get(k, "") for k in self.curve_fields})
            self._buffer.append(row)
            if len(self._buffer) >= self.flush_every:
                self._flush_buffer()
        except Exception as exc:
            print(f"[ExperimentLogger] log_step failed: {exc}")

    def _flush_buffer(self) -> None:
        if not self._buffer or self.curve_path is None:
            return
        with open(self.curve_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.curve_fields)
            writer.writerows(self._buffer)
        self._buffer = []

    def end_run(self, final_metrics: dict) -> None:
        try:
            self._flush_buffer()
            row = {k: "" for k in self.master_fields}
            row.update(self._config)
            row.update(final_metrics)
            with open(self.master_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.master_fields)
                writer.writerow({k: row.get(k, "") for k in self.master_fields})
        except Exception as exc:
            print(f"[ExperimentLogger] end_run failed: {exc}")


def safe_git_hash(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_layer_sizes(model: torch.nn.Module):
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    if not linears:
        return None, None, []
    input_dim = int(linears[0].in_features)
    output_dim = int(linears[-1].out_features)
    hidden = [int(m.out_features) for m in linears[:-1]]
    return input_dim, output_dim, hidden


def activation_name(model: torch.nn.Module) -> str:
    for m in model.modules():
        if isinstance(
            m,
            (
                torch.nn.Tanh,
                torch.nn.ReLU,
                torch.nn.GELU,
                torch.nn.Sigmoid,
                torch.nn.SiLU,
                torch.nn.ELU,
                torch.nn.LeakyReLU,
            ),
        ):
            return m.__class__.__name__.lower()
    return "unknown"


# Legacy architecture-search helpers consolidated here from utils/experiment_logger.py
ARCH_SEARCH_COLUMNS = [
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


def ensure_arch_search_csv(path: str | Path) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ARCH_SEARCH_COLUMNS)
        writer.writeheader()


def append_arch_search_result(path: str | Path, result: dict) -> None:
    ensure_arch_search_csv(path)
    csv_path = Path(path)
    row = {col: result.get(col, "") for col in ARCH_SEARCH_COLUMNS}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ARCH_SEARCH_COLUMNS)
        writer.writerow(row)


def load_best_arch_result(path: str | Path, metric: str = "val_rmse_data") -> dict | None:
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


def compare_arch_to_best(current: dict, best: dict | None) -> tuple[bool, str]:
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
        return False, f"RMSE gain {improvement:.6e} is below threshold {min_required:.6e}."

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
        (
            f"RMSE improved by {improvement:.6e} "
            f"(baseline {best_rmse:.6e} -> current {cur_rmse:.6e}) with PDE guardrail satisfied."
        ),
    )
