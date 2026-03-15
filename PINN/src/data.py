from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split


@dataclass
class PINNBatch:
    # Collocation (PDE residual)
    xi_r: torch.Tensor
    tau_r: torch.Tensor

    # Initial condition (tau=0)
    xi_ic: torch.Tensor
    tau_ic: torch.Tensor
    theta_ic: torch.Tensor

    # Right boundary condition (xi=1) - now flux for Neumann BC
    xi_bc: torch.Tensor
    tau_bc: torch.Tensor
    flux_bc: Optional[torch.Tensor] = None  # nondimensional flux (known mode only)

    # Optional interior data points
    xi_data: Optional[torch.Tensor] = None
    tau_data: Optional[torch.Tensor] = None
    theta_data: Optional[torch.Tensor] = None


@dataclass
class ParamPINNBatch:
    # Collocation (PDE residual)
    xi_r: torch.Tensor
    tau_r: torch.Tensor
    mu_r: torch.Tensor

    # Initial condition (tau=0)
    xi_ic: torch.Tensor
    tau_ic: torch.Tensor
    mu_ic: torch.Tensor
    theta_ic: torch.Tensor

    # Right boundary condition (xi=1) - now flux for Neumann BC
    xi_bc: torch.Tensor
    tau_bc: torch.Tensor
    mu_bc: torch.Tensor
    bc_time_scale: torch.Tensor
    bc_flux_scale: torch.Tensor
    flux_bc: torch.Tensor

    # Optional interior data points
    xi_data: torch.Tensor
    tau_data: torch.Tensor
    mu_data: torch.Tensor
    theta_data: torch.Tensor


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def load_npz_dataset(npz_path: str | Path, device: torch.device) -> PINNBatch:
    """
    Expected keys in the .npz:
      xi_r, tau_r
      xi_ic, tau_ic, theta_ic
      xi_bc, tau_bc, flux_bc (optional for unknown flux mode)
    Optional:
      xi_data, tau_data, theta_data
    """
    data = np.load(npz_path)
    required = ["xi_r", "tau_r", "xi_ic", "tau_ic", "theta_ic", "xi_bc", "tau_bc"]
    for k in required:
        if k not in data:
            raise ValueError(f"Missing key '{k}' in {npz_path}. Found: {list(data.keys())}")

    batch = PINNBatch(
        xi_r=_to_tensor(data["xi_r"], device),
        tau_r=_to_tensor(data["tau_r"], device),
        xi_ic=_to_tensor(data["xi_ic"], device),
        tau_ic=_to_tensor(data["tau_ic"], device),
        theta_ic=_to_tensor(data["theta_ic"], device),
        xi_bc=_to_tensor(data["xi_bc"], device),
        tau_bc=_to_tensor(data["tau_bc"], device),
        flux_bc=_to_tensor(data["flux_bc"], device) if "flux_bc" in data else None,
    )

    if "xi_data" in data and "tau_data" in data and "theta_data" in data:
        batch.xi_data = _to_tensor(data["xi_data"], device)
        batch.tau_data = _to_tensor(data["tau_data"], device)
        batch.theta_data = _to_tensor(data["theta_data"], device)

    return batch


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    """Load manifest rows while preserving JSON commas in q_right_params."""
    manifest_path = Path(manifest_path)
    cols = [
        "case_id",
        "L",
        "k",
        "rho_c",
        "T_left",
        "q_right_type",
        "q_right_params",
        "Nx",
        "t_end",
        "alpha",
        "dx",
        "dt",
        "Nt",
        "mat_path",
        "meta_path",
    ]
    rows: list[dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split(",")
            left = parts[:6]
            right = parts[-8:]
            q_params = ",".join(parts[6:-8])
            row_vals = left + [q_params] + right
            rows.append({k: v for k, v in zip(cols, row_vals)})
    return rows


def _resolve_case_mat_path(case_id: str, mat_path_raw: str, root: Path | None = None) -> Path:
    mat_path = Path(mat_path_raw)
    if mat_path.exists():
        return mat_path
    roots: list[Path] = []
    if root is not None:
        roots.append(root)
    roots.append(Path.cwd())
    for r in roots:
        candidate = r / "data" / "raw" / f"{case_id}.mat"
        if candidate.exists():
            return candidate
    return mat_path

# q_right_values centralizes the logic for computing the right boundary flux values based on the specified type
#  and parameters, allowing for easy extension to new types in the future. It also ensures that the output is properly shaped for later use in the PINN training.
def _q_right_values(q_type: str, q_params: dict[str, Any], t: np.ndarray) -> np.ndarray:
    t1 = t.reshape(-1)
    if q_type == "constant":
        A = float(q_params["A"])
        return A + 0.0 * t1
    if q_type == "sine":
        A = float(q_params["A"])
        omega = float(q_params["omega"])
        phase = float(q_params.get("phase", 0.0))
        return A * np.sin(omega * t1 + phase)
    if q_type == "offset_sine":
        q0 = float(q_params["q0"])
        A = float(q_params["A"])
        omega = float(q_params["omega"])
        return q0 + A * np.sin(omega * t1)
    raise ValueError(f"Unknown q_right_type: {q_type}")

# load_case_manifest_row centralizes the loading and nondimensionalization logic for a single case manifest row, 
# returning a structured dictionary with raw and nondimensional data ready for PINN training. 
# It also includes runtime checks for data consistency and handles various q_right types.
def load_case_manifest_row(row: dict, root: str | Path | None = None) -> dict[str, Any]:
    """Load a manifest case and centralize nondimensionalization.

    Assumptions:
    - T_ref is the left boundary reference temperature `T_left`.
    - Delta_T_ref is max(abs(T - T_ref)) over the loaded case (fallback 1.0).
    - xi = x / L
    - tau = alpha * t / L^2
    - theta = (T - T_ref) / Delta_T_ref
    - Known nondimensional BC target is theta_xi = -q_right / (k * Delta_T_ref / L)
    """
    root_path = Path(root) if root is not None else None
    case_id = str(row["case_id"])
    mat_path = _resolve_case_mat_path(case_id, str(row["mat_path"]), root_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Data file not found for case '{case_id}': {mat_path}")

    mat = sio.loadmat(mat_path)
    T = np.asarray(mat["T"], dtype=np.float32)  # (Nx, Nt)
    x = np.asarray(mat["x"], dtype=np.float32).reshape(-1)  # (Nx,)
    t = np.asarray(mat["time"], dtype=np.float32).reshape(-1)  # (Nt,)

    L = float(row["L"])
    alpha = float(row["alpha"])
    k = float(row["k"])
    T_ref = float(row["T_left"])
    q_type = str(row["q_right_type"])
    q_params = json.loads(str(row["q_right_params"]))

    if not np.isfinite(L) or L <= 0.0:
        raise ValueError(f"Invalid L for case '{case_id}': {L}")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"Invalid alpha for case '{case_id}': {alpha}")

    xi = (x / L).astype(np.float32)
    tau = (alpha * t / (L**2)).astype(np.float32)

    # Runtime safeguard for tau scaling definition.
    tau_expected = (alpha * t / (L**2)).astype(np.float32)
    if not np.allclose(tau, tau_expected, rtol=1e-6, atol=1e-8):
        raise AssertionError("Tau scaling mismatch: expected alpha*t/L^2.")

    theta_raw = T - T_ref
    dT_ref = float(np.max(np.abs(theta_raw)))
    if dT_ref <= 0.0 or not np.isfinite(dT_ref):
        dT_ref = 1.0
    theta = (theta_raw / dT_ref).astype(np.float32)

    q_right = _q_right_values(q_type, q_params, t).astype(np.float32).reshape(-1, 1)
    flux_scale = k * dT_ref / L
    flux_bc_known = (-q_right / flux_scale).astype(np.float32)  # target theta_xi at xi=1

    xi_ic = xi.reshape(-1, 1).astype(np.float32)
    tau_ic = np.zeros_like(xi_ic, dtype=np.float32)
    theta_ic = theta[:, 0].reshape(-1, 1).astype(np.float32)

    tau_bc = tau.reshape(-1, 1).astype(np.float32)
    xi_bc = np.ones_like(tau_bc, dtype=np.float32)

    Ttau, Xxi = np.meshgrid(tau, xi, indexing="ij")
    theta_grid = theta.T  # (Nt, Nx)
    xi_data_all = Xxi.reshape(-1, 1).astype(np.float32)
    tau_data_all = Ttau.reshape(-1, 1).astype(np.float32)
    theta_data_all = theta_grid.reshape(-1, 1).astype(np.float32)

    return {
        "case_id": case_id,
        "manifest_row": dict(row),
        "paths": {"mat_path": str(mat_path), "meta_path": str(row.get("meta_path", ""))},
        "physical": {"L": L, "alpha": alpha, "k": k, "T_ref": T_ref, "dT_ref": dT_ref},
        "raw": {"T": T, "x": x, "t": t, "q_right": q_right},
        "nondim": {"xi": xi, "tau": tau, "theta": theta, "flux_bc_known": flux_bc_known},
        "ic": {"xi_ic": xi_ic, "tau_ic": tau_ic, "theta_ic": theta_ic},
        "bc": {"xi_bc": xi_bc, "tau_bc": tau_bc},
        "interior": {
            "xi_data_all": xi_data_all,
            "tau_data_all": tau_data_all,
            "theta_data_all": theta_data_all,
        },
    }
# build_train_val_batches_from_case centralizes the train/val split logic and allows for known vs unknown flux modes.
def build_train_val_batches_from_case(
    case: dict[str, Any],
    device: torch.device,
    flux_mode: str = "known",
    n_r: int = 50000,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[PINNBatch, PINNBatch]:
    flux_mode = str(flux_mode).lower()
    if flux_mode not in {"known", "unknown"}:
        raise ValueError(f"flux_mode must be 'known' or 'unknown', got: {flux_mode}")

    rng = np.random.default_rng(seed)
    xi = case["nondim"]["xi"]
    tau = case["nondim"]["tau"]
    flux_bc_known = case["nondim"]["flux_bc_known"]
    xi_ic = case["ic"]["xi_ic"]
    tau_ic = case["ic"]["tau_ic"]
    theta_ic = case["ic"]["theta_ic"]
    xi_bc = case["bc"]["xi_bc"]
    tau_bc = case["bc"]["tau_bc"]
    xi_data_all = case["interior"]["xi_data_all"]
    tau_data_all = case["interior"]["tau_data_all"]
    theta_data_all = case["interior"]["theta_data_all"]

    n_total = len(xi_data_all)
    indices = np.arange(n_total)
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed)

    xi_r = rng.uniform(0.0, 1.0, size=(n_r, 1)).astype(np.float32)
    tau_r = rng.uniform(0.0, float(tau.max()), size=(n_r, 1)).astype(np.float32)
    xi_r_val = rng.uniform(0.0, 1.0, size=(max(1, n_r // 5), 1)).astype(np.float32)
    tau_r_val = rng.uniform(0.0, float(tau.max()), size=(max(1, n_r // 5), 1)).astype(np.float32)

    flux_train: Optional[torch.Tensor] = None
    flux_val: Optional[torch.Tensor] = None
    if flux_mode == "known":
        known_flux_tensor = torch.tensor(flux_bc_known, dtype=torch.float32, device=device)
        flux_train = known_flux_tensor
        flux_val = known_flux_tensor

    train_batch = PINNBatch(
        xi_r=torch.tensor(xi_r, dtype=torch.float32, device=device),
        tau_r=torch.tensor(tau_r, dtype=torch.float32, device=device),
        xi_ic=torch.tensor(xi_ic, dtype=torch.float32, device=device),
        tau_ic=torch.tensor(tau_ic, dtype=torch.float32, device=device),
        theta_ic=torch.tensor(theta_ic, dtype=torch.float32, device=device),
        xi_bc=torch.tensor(xi_bc, dtype=torch.float32, device=device),
        tau_bc=torch.tensor(tau_bc, dtype=torch.float32, device=device),
        flux_bc=flux_train,
        xi_data=torch.tensor(xi_data_all[train_idx], dtype=torch.float32, device=device),
        tau_data=torch.tensor(tau_data_all[train_idx], dtype=torch.float32, device=device),
        theta_data=torch.tensor(theta_data_all[train_idx], dtype=torch.float32, device=device),
    )
    val_batch = PINNBatch(
        xi_r=torch.tensor(xi_r_val, dtype=torch.float32, device=device),
        tau_r=torch.tensor(tau_r_val, dtype=torch.float32, device=device),
        xi_ic=torch.tensor(xi_ic, dtype=torch.float32, device=device),
        tau_ic=torch.tensor(tau_ic, dtype=torch.float32, device=device),
        theta_ic=torch.tensor(theta_ic, dtype=torch.float32, device=device),
        xi_bc=torch.tensor(xi_bc, dtype=torch.float32, device=device),
        tau_bc=torch.tensor(tau_bc, dtype=torch.float32, device=device),
        flux_bc=flux_val,
        xi_data=torch.tensor(xi_data_all[val_idx], dtype=torch.float32, device=device),
        tau_data=torch.tensor(tau_data_all[val_idx], dtype=torch.float32, device=device),
        theta_data=torch.tensor(theta_data_all[val_idx], dtype=torch.float32, device=device),
    )
    return train_batch, val_batch


def sample_uniform(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low=low, high=high, size=(n, 1)).astype(np.float32)

# build_synthetic_batch allows for on-the-fly sampling of collocation, IC, BC, and 
# optional interior data points in the nondimensional domain, using user-provided functions for IC and BC values.
#  This is useful for synthetic experiments where you have an analytical solution or want to test specific scenarios without relying on pre-saved datasets.
def build_synthetic_batch(
    n_r: int,
    n_ic: int,
    n_bc: int,
    tau_max: float,
    device: torch.device,
    ic_fn: Callable[[np.ndarray], np.ndarray],
    bc_right_flux_fn: Callable[[np.ndarray], np.ndarray],  # now flux function
    n_data: int = 0,
    data_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    seed: int = 42,
) -> PINNBatch:
    """
    On-the-fly sampling in nondimensional domain:
      xi in [0,1], tau in [0,tau_max]
    ic_fn: theta(xi) at tau=0
    bc_right_flux_fn: flux(tau) at xi=1 (nondimensional)
    data_fn: theta(xi,tau) interior (optional) if you have synthetic "truth"
    """
    rng = np.random.default_rng(seed)

    # Collocation points
    xi_r = sample_uniform(n_r, 0.0, 1.0, rng)
    tau_r = sample_uniform(n_r, 0.0, tau_max, rng)

    # IC points (tau=0)
    xi_ic = sample_uniform(n_ic, 0.0, 1.0, rng)
    tau_ic = np.zeros((n_ic, 1), dtype=np.float32)
    theta_ic = ic_fn(xi_ic).astype(np.float32)

    # Right BC points (xi=1)
    xi_bc = np.ones((n_bc, 1), dtype=np.float32)
    tau_bc = sample_uniform(n_bc, 0.0, tau_max, rng)
    flux_bc = bc_right_flux_fn(tau_bc).astype(np.float32)

    batch = PINNBatch(
        xi_r=torch.tensor(xi_r, dtype=torch.float32, device=device),
        tau_r=torch.tensor(tau_r, dtype=torch.float32, device=device),
        xi_ic=torch.tensor(xi_ic, dtype=torch.float32, device=device),
        tau_ic=torch.tensor(tau_ic, dtype=torch.float32, device=device),
        theta_ic=torch.tensor(theta_ic, dtype=torch.float32, device=device),
        xi_bc=torch.tensor(xi_bc, dtype=torch.float32, device=device),
        tau_bc=torch.tensor(tau_bc, dtype=torch.float32, device=device),
        flux_bc=torch.tensor(flux_bc, dtype=torch.float32, device=device),
    )

    if n_data > 0:
        if data_fn is None:
            raise ValueError("n_data>0 requires data_fn(xi,tau) for interior theta.")
        xi_data = sample_uniform(n_data, 0.0, 1.0, rng)
        tau_data = sample_uniform(n_data, 0.0, tau_max, rng)
        theta_data = data_fn(xi_data, tau_data).astype(np.float32)

        batch.xi_data = torch.tensor(xi_data, dtype=torch.float32, device=device)
        batch.tau_data = torch.tensor(tau_data, dtype=torch.float32, device=device)
        batch.theta_data = torch.tensor(theta_data, dtype=torch.float32, device=device)

    return batch
