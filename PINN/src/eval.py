from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import Scales, ensure_dir
from .pinn import FluxLearnedConfig, dtheta_dxi, interp1d_linear, predict_theta


def load_model_weights(model: nn.Module, ckpt_path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


def _expand_mu(mu: Optional[np.ndarray | torch.Tensor], n: int, device: torch.device) -> Optional[torch.Tensor]:
    if mu is None:
        return None
    if isinstance(mu, torch.Tensor):
        mu_t = mu.to(device=device, dtype=torch.float32)
    else:
        mu_t = torch.tensor(np.asarray(mu), dtype=torch.float32, device=device)
    if mu_t.ndim == 1:
        mu_t = mu_t.unsqueeze(0)
    if mu_t.shape[0] == 1:
        mu_t = mu_t.repeat(n, 1)
    if mu_t.shape[0] != n:
        raise ValueError(f"mu must have batch dimension 1 or {n}, got {tuple(mu_t.shape)}")
    return mu_t


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    tau_vals: np.ndarray,
    xi_vals: np.ndarray,
    device: torch.device,
    mu: Optional[np.ndarray | torch.Tensor] = None,
) -> np.ndarray:
    """Return theta grid of shape (len(tau_vals), len(xi_vals))."""
    T, X = np.meshgrid(tau_vals.astype(np.float32), xi_vals.astype(np.float32), indexing="ij")
    xi = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    tau = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    mu_t = _expand_mu(mu, xi.shape[0], device)
    theta = predict_theta(model, xi, tau, mu_t).cpu().numpy().reshape(len(tau_vals), len(xi_vals))
    return theta


def recover_flux_vs_tau(
    model: nn.Module,
    tau_vals: np.ndarray,
    device: torch.device,
    scales: Scales,
    xi0: float = 1.0,
    mu: Optional[np.ndarray | torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (tau_vals, q_flux) with optional parametric conditioning."""
    tau_t = torch.tensor(tau_vals.reshape(-1, 1), dtype=torch.float32, device=device)
    xi_t = torch.full_like(tau_t, float(xi0), dtype=torch.float32, device=device)
    mu_t = _expand_mu(mu, tau_t.shape[0], device)

    theta_xi = dtheta_dxi(model, xi_t, tau_t, mu_t)
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


@torch.no_grad()
def predict_q_hat(
    tau_vals: np.ndarray,
    flux_cfg: FluxLearnedConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict learned nondimensional flux q_hat(tau) on a grid."""
    tau_t = torch.tensor(tau_vals.reshape(-1, 1), dtype=torch.float32, device=device)
    q_hat = interp1d_linear(flux_cfg.tau_knots, flux_cfg.q_ctrl, tau_t)
    return tau_vals, q_hat.detach().cpu().numpy().reshape(-1)
