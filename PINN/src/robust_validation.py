from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from .data import load_case_manifest_row, load_manifest_rows
from .pinn import dtheta_dxi, pde_residual_theta_tau_minus_theta_xi_xx, predict_theta
from .utils import ensure_dir, save_json


@dataclass
class SplitBundle:
    train_rows: list[dict[str, str]]
    val_rows: list[dict[str, str]]
    test_rows: list[dict[str, str]]
    ood_rows: list[dict[str, str]]


@dataclass
class ObservationConfig:
    sensor_indices: list[int] | None = None
    sensor_mode: str = "nearest_right"
    time_stride: int = 1
    mu_time_samples: int = 15


@dataclass
class NoiseConfig:
    name: str = "clean"
    gaussian_std: float = 0.0
    constant_bias: float = 0.0
    drift_final: float = 0.0
    temporal_corr: int = 0
    spatial_corr: int = 0
    spike_fraction: float = 0.0
    spike_scale: float = 0.0
    missing_fraction: float = 0.0
    missing_axis: str = "time"
    dead_sensor_fraction: float = 0.0
    sensor_noise_std_scale: float = 0.0


def _case_params(row_or_case: dict[str, Any]) -> dict[str, Any]:
    if "manifest_row" in row_or_case:
        return json.loads(str(row_or_case["manifest_row"]["q_right_params"]))
    return json.loads(str(row_or_case["q_right_params"]))


def flux_family(row_or_case: dict[str, Any]) -> str:
    q_type = str(row_or_case.get("q_right_type", row_or_case.get("manifest_row", {}).get("q_right_type", ""))).lower()
    params = _case_params(row_or_case)
    if q_type == "sine":
        return "sine"
    if q_type == "offset_sine":
        return "offset_sine"
    if q_type == "constant":
        return "constant"
    return q_type or "unknown"


def flux_signature(row_or_case: dict[str, Any]) -> dict[str, float | str]:
    q_type = flux_family(row_or_case)
    params = _case_params(row_or_case)
    amp = float(params.get("A", params.get("q0", 0.0)))
    omega = float(params.get("omega", 0.0))
    return {"family": q_type, "amplitude": amp, "omega": omega}


def build_case_splits(
    manifest_train: str | Path,
    manifest_testing: str | Path,
    val_fraction: float = 0.2,
) -> SplitBundle:
    rows_train_all = load_manifest_rows(manifest_train)
    rows_testing = load_manifest_rows(manifest_testing)
    if len(rows_train_all) < 2:
        raise RuntimeError("Need at least two cases for train/held-out split.")

    # Deterministic split; keep final val_fraction as held-out.
    rows_sorted = sorted(rows_train_all, key=lambda r: r["case_id"])
    n_val = max(1, int(round(val_fraction * len(rows_sorted))))
    val_rows = rows_sorted[-n_val:]
    train_rows = rows_sorted[:-n_val]
    if len(train_rows) == 0:
        raise RuntimeError("Split produced no training cases.")

    train_signatures = [flux_signature(r) for r in train_rows]
    train_families = {sig["family"] for sig in train_signatures}
    amp_by_family: dict[str, tuple[float, float]] = {}
    omega_by_family: dict[str, tuple[float, float]] = {}
    for fam in train_families:
        fam_sigs = [sig for sig in train_signatures if sig["family"] == fam]
        amp_by_family[fam] = (
            min(float(sig["amplitude"]) for sig in fam_sigs),
            max(float(sig["amplitude"]) for sig in fam_sigs),
        )
        omega_by_family[fam] = (
            min(float(sig["omega"]) for sig in fam_sigs),
            max(float(sig["omega"]) for sig in fam_sigs),
        )

    test_rows: list[dict[str, str]] = []
    ood_rows: list[dict[str, str]] = []
    for row in rows_testing:
        sig = flux_signature(row)
        fam = str(sig["family"])
        amp = float(sig["amplitude"])
        omega = float(sig["omega"])
        fam_known = fam in train_families
        amp_range = amp_by_family.get(fam, (math.inf, -math.inf))
        omega_range = omega_by_family.get(fam, (math.inf, -math.inf))
        out_of_range = amp < amp_range[0] or amp > amp_range[1] or omega < omega_range[0] or omega > omega_range[1]
        if (not fam_known) or out_of_range:
            ood_rows.append(row)
        else:
            test_rows.append(row)

    return SplitBundle(train_rows=train_rows, val_rows=val_rows, test_rows=test_rows, ood_rows=ood_rows)


def leave_one_flux_family_out(rows: Iterable[dict[str, str]]) -> dict[str, SplitBundle]:
    rows_list = list(rows)
    families = sorted({flux_family(r) for r in rows_list})
    out: dict[str, SplitBundle] = {}
    for family in families:
        held_out = [r for r in rows_list if flux_family(r) == family]
        kept = [r for r in rows_list if flux_family(r) != family]
        out[family] = SplitBundle(train_rows=kept, val_rows=[], test_rows=held_out, ood_rows=held_out)
    return out


def compute_mu_stats(cases: list[dict[str, Any]]) -> dict[str, np.ndarray | float]:
    eps = 1e-8
    all_mu = np.array([c["mu_raw"] for c in cases], dtype=np.float32)
    return {
        "mu_min": all_mu.min(axis=0),
        "mu_max": all_mu.max(axis=0),
        "eps": eps,
    }


def normalise_mu(mu_raw: np.ndarray, mu_stats: dict[str, np.ndarray | float]) -> np.ndarray:
    eps = float(mu_stats["eps"])
    mu_min = np.asarray(mu_stats["mu_min"], dtype=np.float32)
    mu_max = np.asarray(mu_stats["mu_max"], dtype=np.float32)
    return 2.0 * (mu_raw - mu_min) / (mu_max - mu_min + eps) - 1.0


def _build_mu_raw_from_trace(theta_trace: np.ndarray, tau: np.ndarray, mu_time_samples: int) -> np.ndarray:
    tau_samples = np.linspace(float(tau.min()), float(tau.max()), mu_time_samples, dtype=np.float32)
    theta_series = np.interp(tau_samples, tau, theta_trace).astype(np.float32)
    dtheta_dtau = np.gradient(theta_series, tau_samples).astype(np.float32)
    return np.concatenate([theta_series, dtheta_dtau]).astype(np.float32)


def load_case_with_mu(
    row: dict[str, str],
    root: str | Path | None,
    mu_time_samples: int = 15,
) -> dict[str, Any]:
    case = load_case_manifest_row(row, root=root)
    theta = case["nondim"]["theta"]
    tau = case["nondim"]["tau"]
    mu_raw = _build_mu_raw_from_trace(theta[-1, :], tau, mu_time_samples=mu_time_samples)
    case["mu_raw"] = mu_raw
    case["mu"] = mu_raw.copy()
    return case


def load_cases_with_mu(
    rows: Iterable[dict[str, str]],
    root: str | Path | None,
    mu_time_samples: int = 15,
) -> list[dict[str, Any]]:
    return [load_case_with_mu(r, root=root, mu_time_samples=mu_time_samples) for r in rows]


def _smooth_along_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return values
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= kernel.sum()
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=axis, arr=values)


def apply_temperature_noise(
    theta: np.ndarray,
    cfg: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy = np.array(theta, dtype=np.float32, copy=True)
    nx, nt = noisy.shape
    if cfg.gaussian_std > 0.0:
        noisy += rng.normal(0.0, cfg.gaussian_std, size=noisy.shape).astype(np.float32)
    if cfg.constant_bias != 0.0:
        noisy += np.float32(cfg.constant_bias)
    if cfg.drift_final != 0.0:
        noisy += np.linspace(0.0, cfg.drift_final, nt, dtype=np.float32)[None, :]
    if cfg.sensor_noise_std_scale > 0.0:
        per_sensor = rng.uniform(
            1.0 - cfg.sensor_noise_std_scale,
            1.0 + cfg.sensor_noise_std_scale,
            size=(nx, 1),
        ).astype(np.float32)
        noisy += per_sensor * rng.normal(0.0, cfg.gaussian_std, size=noisy.shape).astype(np.float32)
    if cfg.temporal_corr > 0:
        noisy = _smooth_along_axis(noisy, cfg.temporal_corr, axis=1)
    if cfg.spatial_corr > 0:
        noisy = _smooth_along_axis(noisy, cfg.spatial_corr, axis=0)
    if cfg.spike_fraction > 0.0 and cfg.spike_scale > 0.0:
        n_spikes = max(1, int(round(cfg.spike_fraction * noisy.size)))
        spike_idx = rng.choice(noisy.size, size=n_spikes, replace=False)
        flat = noisy.reshape(-1)
        flat[spike_idx] += rng.normal(0.0, cfg.spike_scale, size=n_spikes).astype(np.float32)
        noisy = flat.reshape(noisy.shape)
    if cfg.missing_fraction > 0.0:
        if cfg.missing_axis == "time":
            count = max(1, int(round(cfg.missing_fraction * nt)))
            idx = rng.choice(nt, size=count, replace=False)
            noisy[:, idx] = np.nan
        else:
            count = max(1, int(round(cfg.missing_fraction * nx)))
            idx = rng.choice(nx, size=count, replace=False)
            noisy[idx, :] = np.nan
    if cfg.dead_sensor_fraction > 0.0:
        count = max(1, int(round(cfg.dead_sensor_fraction * nx)))
        idx = rng.choice(nx, size=count, replace=False)
        noisy[idx, :] = np.nan
    return noisy


def impute_missing_theta(theta_obs: np.ndarray, tau: np.ndarray) -> np.ndarray:
    out = np.array(theta_obs, dtype=np.float32, copy=True)
    for i in range(out.shape[0]):
        row = out[i]
        mask = np.isfinite(row)
        if not mask.any():
            out[i, :] = 0.0
            continue
        if mask.all():
            continue
        out[i, :] = np.interp(tau, tau[mask], row[mask]).astype(np.float32)
    return out


def select_sensor_indices(xi: np.ndarray, mode: str, sensor_count: Optional[int], rng: np.random.Generator) -> list[int]:
    n = len(xi)
    if sensor_count is None or sensor_count >= n:
        return list(range(n))
    mode = str(mode).lower()
    if mode == "boundary_only":
        candidates = [0, n - 1]
    elif mode == "interior_only":
        candidates = list(range(1, max(1, n - 1)))
    elif mode == "random":
        candidates = list(range(n))
    else:
        candidates = list(range(n))
    if len(candidates) <= sensor_count:
        return sorted(candidates)
    return sorted(rng.choice(candidates, size=sensor_count, replace=False).tolist())


def build_observation_trace(
    theta_obs: np.ndarray,
    xi: np.ndarray,
    tau: np.ndarray,
    cfg: ObservationConfig,
) -> np.ndarray:
    sensor_indices = cfg.sensor_indices if cfg.sensor_indices is not None else list(range(len(xi)))
    trace_bank = theta_obs[sensor_indices, :]
    sensor_mode = str(cfg.sensor_mode).lower()
    if trace_bank.ndim == 1:
        trace_bank = trace_bank[None, :]
    if sensor_mode == "mean":
        trace = np.nanmean(trace_bank, axis=0)
    else:
        selected_xi = xi[np.array(sensor_indices)]
        best_idx = int(np.argmax(selected_xi))
        trace = trace_bank[best_idx, :]
    trace = np.asarray(trace, dtype=np.float32)
    if np.isnan(trace).any():
        mask = np.isfinite(trace)
        trace = np.interp(tau, tau[mask], trace[mask]).astype(np.float32)
    if cfg.time_stride > 1:
        tau_sparse = tau[:: cfg.time_stride]
        trace_sparse = trace[:: cfg.time_stride]
        trace = np.interp(tau, tau_sparse, trace_sparse).astype(np.float32)
    return trace


def make_case_observation(
    case: dict[str, Any],
    mu_stats: dict[str, np.ndarray | float],
    obs_cfg: ObservationConfig,
    noise_cfg: NoiseConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    xi = case["nondim"]["xi"]
    tau = case["nondim"]["tau"]
    theta_true = case["nondim"]["theta"]
    theta_noisy = apply_temperature_noise(theta_true, noise_cfg, rng)
    theta_noisy = impute_missing_theta(theta_noisy, tau)
    trace = build_observation_trace(theta_noisy, xi, tau, obs_cfg)
    mu_raw = _build_mu_raw_from_trace(trace, tau, mu_time_samples=obs_cfg.mu_time_samples)
    return {
        "theta_observed": theta_noisy,
        "trace": trace,
        "mu_raw": mu_raw,
        "mu": normalise_mu(mu_raw, mu_stats),
    }


def predict_case_from_observation(
    model: torch.nn.Module,
    case: dict[str, Any],
    observation: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    xi = case["nondim"]["xi"]
    tau = case["nondim"]["tau"]
    theta_true = case["nondim"]["theta"]
    mu_case = np.asarray(observation["mu"], dtype=np.float32)

    xi_grid, tau_grid = np.meshgrid(xi, tau, indexing="ij")
    X_pred = np.column_stack(
        [xi_grid.reshape(-1), tau_grid.reshape(-1), np.tile(mu_case, (xi_grid.size, 1))]
    ).astype(np.float32)
    X_tensor = torch.tensor(X_pred, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True)

    theta_pred_flat = model(X_tensor)
    theta_pred = theta_pred_flat.detach().cpu().numpy().reshape(xi_grid.shape)

    grads = torch.autograd.grad(
        outputs=theta_pred_flat,
        inputs=X_tensor,
        grad_outputs=torch.ones_like(theta_pred_flat),
        create_graph=False,
        retain_graph=False,
    )[0]
    theta_xi = grads[:, 0].detach().cpu().numpy().reshape(xi_grid.shape)

    phys = case["physical"]
    q_true = np.asarray(case["raw"]["q_right"], dtype=np.float32).reshape(-1)
    flux_scale = float(phys["k"]) * float(phys["dT_ref"]) / float(phys["L"])
    q_pred = (-flux_scale * theta_xi[-1, :]).astype(np.float32)

    T_left = float(phys["T_ref"])
    dT = float(phys["dT_ref"])
    T_true = T_left + dT * theta_true
    T_pred = T_left + dT * theta_pred

    return {
        "case_id": case["case_id"],
        "xi": xi,
        "tau": tau,
        "theta_true": theta_true,
        "theta_pred": theta_pred,
        "theta_observed": observation["theta_observed"],
        "observation_trace": observation["trace"],
        "mu": mu_case,
        "q_true": q_true,
        "q_pred": q_pred,
        "T_true": T_true,
        "T_pred": T_pred,
        "theta_xi_pred": theta_xi,
        "flux_scale": flux_scale,
        "q_type": flux_family(case),
        "q_params": _case_params(case),
    }


def compute_flux_metrics(q_true: np.ndarray, q_pred: np.ndarray, tau: np.ndarray) -> dict[str, float]:
    q_true = np.asarray(q_true, dtype=np.float64).reshape(-1)
    q_pred = np.asarray(q_pred, dtype=np.float64).reshape(-1)
    diff = q_pred - q_true
    denom = np.linalg.norm(q_true) + 1e-12
    true_peak_idx = int(np.argmax(np.abs(q_true)))
    pred_peak_idx = int(np.argmax(np.abs(q_pred)))
    corr = np.corrcoef(q_true, q_pred)[0, 1] if np.std(q_true) > 0 and np.std(q_pred) > 0 else math.nan
    return {
        "flux_rmse": float(np.sqrt(np.mean(diff**2))),
        "flux_mae": float(np.mean(np.abs(diff))),
        "flux_rel_l2": float(np.linalg.norm(diff) / denom),
        "flux_max_abs": float(np.max(np.abs(diff))),
        "peak_flux_error": float(abs(q_pred[true_peak_idx] - q_true[true_peak_idx])),
        "time_to_peak_error": float(abs(float(tau[pred_peak_idx]) - float(tau[true_peak_idx]))),
        "integrated_flux_error": float(abs(np.trapz(q_pred, tau) - np.trapz(q_true, tau))),
        "flux_corrcoef": float(corr) if np.isfinite(corr) else math.nan,
    }


def compute_temperature_metrics(theta_true: np.ndarray, theta_pred: np.ndarray) -> dict[str, float]:
    diff = np.asarray(theta_pred - theta_true, dtype=np.float64)
    denom = np.linalg.norm(theta_true) + 1e-12
    return {
        "theta_rmse": float(np.sqrt(np.mean(diff**2))),
        "theta_rel_l2": float(np.linalg.norm(diff) / denom),
    }


def compute_boundary_profile_metrics(theta_true: np.ndarray, theta_pred: np.ndarray) -> dict[str, float]:
    left_diff = theta_pred[0, :] - theta_true[0, :]
    right_diff = theta_pred[-1, :] - theta_true[-1, :]
    return {
        "left_boundary_rmse": float(np.sqrt(np.mean(left_diff**2))),
        "right_boundary_rmse": float(np.sqrt(np.mean(right_diff**2))),
        "left_boundary_max_abs": float(np.max(np.abs(theta_pred[0, :]))),
    }


def compute_physics_checks(
    model: torch.nn.Module,
    prediction: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    xi = prediction["xi"]
    tau = prediction["tau"]
    mu = prediction["mu"]
    xi_grid, tau_grid = np.meshgrid(xi, tau, indexing="ij")
    sample = np.column_stack(
        [xi_grid.reshape(-1), tau_grid.reshape(-1), np.tile(mu, (xi_grid.size, 1))]
    ).astype(np.float32)
    sample_t = torch.tensor(sample, dtype=torch.float32, device=device)
    xi_t = sample_t[:, :1]
    tau_t = sample_t[:, 1:2]
    mu_t = sample_t[:, 2:]

    residual = pde_residual_theta_tau_minus_theta_xi_xx(
        model,
        xi_t,
        tau_t,
        mu_t,
        create_graph=False,
    )
    residual_np = residual.detach().cpu().numpy().reshape(xi_grid.shape)

    theta_pred = np.asarray(prediction["theta_pred"], dtype=np.float64)
    theta_tau = np.gradient(theta_pred, tau, axis=1)
    stored_energy = np.trapz(theta_pred, xi, axis=0)
    energy_rate = np.gradient(stored_energy, tau)
    left_grad = np.gradient(theta_pred[:, :], xi, axis=0)[0, :]
    right_grad = np.gradient(theta_pred[:, :], xi, axis=0)[-1, :]
    energy_balance = energy_rate - (right_grad - left_grad)

    q_pred = np.asarray(prediction["q_pred"], dtype=np.float64)
    if q_pred.size >= 3:
        second_diff = np.diff(q_pred, n=2)
        smoothness = float(np.mean(np.abs(second_diff)))
    else:
        smoothness = math.nan

    return {
        "pde_residual_rmse": float(np.sqrt(np.mean(residual_np**2))),
        "pde_residual_max_abs": float(np.max(np.abs(residual_np))),
        "energy_balance_rmse": float(np.sqrt(np.mean(energy_balance**2))),
        "flux_smoothness_second_diff": smoothness,
        "nonphysical_flux_oscillation_ratio": float(np.max(np.abs(np.diff(q_pred))) / (np.max(np.abs(q_pred)) + 1e-12)),
    }


def fit_linear_flux_baseline(train_predictions: list[dict[str, Any]], ridge: float = 1e-6) -> dict[str, np.ndarray]:
    X = np.stack([np.concatenate([p["mu"], [1.0]]).astype(np.float64) for p in train_predictions], axis=0)
    Y = np.stack([np.asarray(p["q_true"], dtype=np.float64) for p in train_predictions], axis=0)
    xtx = X.T @ X
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    w = np.linalg.solve(xtx, X.T @ Y)
    return {"weights": w}


def predict_linear_flux_baseline(model: dict[str, np.ndarray], mu: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.asarray(mu, dtype=np.float64), [1.0]], axis=0)
    return (x @ model["weights"]).astype(np.float32)


def summarize_ensemble_flux(
    q_true: np.ndarray,
    q_preds: np.ndarray,
    z_value: float = 1.96,
) -> dict[str, float | np.ndarray]:
    q_preds = np.asarray(q_preds, dtype=np.float64)
    mean = q_preds.mean(axis=0)
    std = q_preds.std(axis=0, ddof=0)
    lower = mean - z_value * std
    upper = mean + z_value * std
    q_true = np.asarray(q_true, dtype=np.float64)
    coverage = np.mean((q_true >= lower) & (q_true <= upper))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "lower": lower.astype(np.float32),
        "upper": upper.astype(np.float32),
        "coverage_95": float(coverage),
    }


def evaluate_case(
    model: torch.nn.Module,
    case: dict[str, Any],
    mu_stats: dict[str, np.ndarray | float],
    device: torch.device,
    obs_cfg: ObservationConfig,
    noise_cfg: NoiseConfig,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observation = make_case_observation(case, mu_stats, obs_cfg, noise_cfg, rng)
    prediction = predict_case_from_observation(model, case, observation, device)
    metrics = {}
    metrics.update(compute_flux_metrics(prediction["q_true"], prediction["q_pred"], prediction["tau"]))
    metrics.update(compute_temperature_metrics(prediction["theta_true"], prediction["theta_pred"]))
    metrics.update(compute_boundary_profile_metrics(prediction["theta_true"], prediction["theta_pred"]))
    metrics.update(compute_physics_checks(model, prediction, device))
    return {
        "prediction": prediction,
        "observation": observation,
        "metrics": metrics,
        "noise": asdict(noise_cfg),
        "observation_cfg": asdict(obs_cfg),
    }


def evaluate_cases(
    model: torch.nn.Module,
    cases: list[dict[str, Any]],
    mu_stats: dict[str, np.ndarray | float],
    device: torch.device,
    obs_cfg: ObservationConfig,
    noise_cfg: NoiseConfig,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    for idx, case in enumerate(cases):
        results.append(
            evaluate_case(
                model,
                case,
                mu_stats,
                device,
                obs_cfg=obs_cfg,
                noise_cfg=noise_cfg,
                seed=seed + idx,
            )
        )
    return results


def results_to_frame(results: list[dict[str, Any]], split_name: str, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "case_id": result["prediction"]["case_id"],
            "split": split_name,
            "model_name": model_name,
            "noise_name": result["noise"]["name"],
        }
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def run_noise_campaign(
    model: torch.nn.Module,
    cases: list[dict[str, Any]],
    mu_stats: dict[str, np.ndarray | float],
    device: torch.device,
    obs_cfg: ObservationConfig,
    noise_scenarios: list[NoiseConfig],
    base_seed: int = 42,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, scenario in enumerate(noise_scenarios):
        results = evaluate_cases(
            model,
            cases,
            mu_stats,
            device,
            obs_cfg=obs_cfg,
            noise_cfg=scenario,
            seed=base_seed + 1000 * idx,
        )
        frames.append(results_to_frame(results, split_name="campaign", model_name="pinn"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_sensor_study(
    model: torch.nn.Module,
    cases: list[dict[str, Any]],
    mu_stats: dict[str, np.ndarray | float],
    device: torch.device,
    sensor_modes: list[tuple[str, int | None]],
    noise_cfg: NoiseConfig,
    mu_time_samples: int,
    base_seed: int = 42,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    xi = cases[0]["nondim"]["xi"]
    for idx, (mode, count) in enumerate(sensor_modes):
        rng = np.random.default_rng(base_seed + 500 * idx)
        sensor_indices = select_sensor_indices(xi, mode=mode, sensor_count=count, rng=rng)
        obs_cfg = ObservationConfig(
            sensor_indices=sensor_indices,
            sensor_mode="nearest_right",
            time_stride=1,
            mu_time_samples=mu_time_samples,
        )
        results = evaluate_cases(
            model,
            cases,
            mu_stats,
            device,
            obs_cfg=obs_cfg,
            noise_cfg=noise_cfg,
            seed=base_seed + 500 * idx,
        )
        frame = results_to_frame(results, split_name="sensor_study", model_name="pinn")
        frame["sensor_selection_mode"] = mode
        frame["sensor_count"] = len(sensor_indices)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def save_experiment_log(
    out_dir: str | Path,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    out_dir = ensure_dir(out_dir)
    path = out_dir / filename
    save_json(payload, path)
    return path


def bundle_metrics_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in frame.columns if c not in {"case_id", "split", "model_name", "noise_name"} and np.issubdtype(frame[c].dtype, np.number)]
    grouped = frame.groupby(group_cols)[metric_cols]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=0).add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()
