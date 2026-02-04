from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import ensure_dir, Scales, save_json
from .pinn import dtheta_dxi, predict_theta


def load_model_weights(model: nn.Module, ckpt_path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    tau_vals: np.ndarray,
    xi_vals: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Returns theta grid of shape (len(tau_vals), len(xi_vals))
    """
    T, X = np.meshgrid(tau_vals.astype(np.float32), xi_vals.astype(np.float32), indexing="ij")
    xi = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    tau = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    theta = predict_theta(model, xi, tau).cpu().numpy().reshape(len(tau_vals), len(xi_vals))
    return theta


def recover_flux_vs_tau(
    model: nn.Module,
    tau_vals: np.ndarray,
    device: torch.device,
    scales: Scales,
    xi0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (tau_vals, q_flux) where:
      q_flux = -k*(dT/L)*dtheta/dxi at xi=0
    """
    tau_t = torch.tensor(tau_vals.reshape(-1, 1), dtype=torch.float32, device=device)
    xi_t = torch.full_like(tau_t, float(xi0), dtype=torch.float32, device=device)

    # Need grads -> do NOT use no_grad
    theta_xi = dtheta_dxi(model, xi_t, tau_t)  # shape (N,1)
    q = -scales.flux_scale * theta_xi.detach().cpu().numpy().reshape(-1)
    return tau_vals, q


def save_flux_csv(out_path: str | Path, tau: np.ndarray, q: np.ndarray) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    arr = np.column_stack([tau, q])
    np.savetxt(out_path, arr, delimiter=",", header="tau,q_flux", comments="")


def save_theta_grid_npz(out_path: str | Path, tau: np.ndarray, xi: np.ndarray, theta: np.ndarray) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    np.savez(out_path, tau=tau, xi=xi, theta=theta)
