from __future__ import annotations

import csv
import json
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
