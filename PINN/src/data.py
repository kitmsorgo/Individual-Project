from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch


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
    flux_bc: torch.Tensor  # nondimensional flux

    # Optional interior data points
    xi_data: Optional[torch.Tensor] = None
    tau_data: Optional[torch.Tensor] = None
    theta_data: Optional[torch.Tensor] = None


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def load_npz_dataset(npz_path: str | Path, device: torch.device) -> PINNBatch:
    """
    Expected keys in the .npz:
      xi_r, tau_r
      xi_ic, tau_ic, theta_ic
      xi_bc, tau_bc, theta_bc
    Optional:
      xi_data, tau_data, theta_data
    """
    data = np.load(npz_path)
    required = ["xi_r", "tau_r", "xi_ic", "tau_ic", "theta_ic", "xi_bc", "tau_bc", "flux_bc"]
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
        flux_bc=_to_tensor(data["flux_bc"], device),
    )

    if "xi_data" in data and "tau_data" in data and "theta_data" in data:
        batch.xi_data = _to_tensor(data["xi_data"], device)
        batch.tau_data = _to_tensor(data["tau_data"], device)
        batch.theta_data = _to_tensor(data["theta_data"], device)

    return batch


def sample_uniform(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low=low, high=high, size=(n, 1)).astype(np.float32)


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
