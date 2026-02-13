from __future__ import annotations

"""PINN model utilities.

This module provides a lightweight fully-connected MLP used as the neural
approximation for the PINN, helpers to compute derivatives via autograd, the
PDE residual for a nondimensional 1D heat equation, and the aggregate loss
computation used by training scripts.

The functions expect a `batch` object (from `src.data`) providing tensors
such as `xi_r`, `tau_r`, `xi_ic`, `tau_ic`, `theta_ic`, etc.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 30, layers: int = 2, out_dim: int = 1):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be >= 2")

        """Simple fully-connected MLP with Tanh activations.

        Architecture: Linear(in_dim -> hidden) + Tanh + (layers-2) * [Linear(hidden->hidden)+Tanh]
        + Linear(hidden->out_dim).
        """

        net = []
        net.append(nn.Linear(in_dim, hidden))
        net.append(nn.Tanh())

        for _ in range(layers - 2):
            net.append(nn.Linear(hidden, hidden))
            net.append(nn.Tanh())

        net.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*net)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Inputs:
        - `x`: tensor shaped (N, in_dim) containing concatenated spatial and temporal inputs.

        Returns predicted scalar field (N, out_dim).
        """
        return self.net(x)


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute gradient of `outputs` w.r.t. `inputs` using PyTorch autograd.

    Returns tensor with same shape as `inputs` representing d(outputs)/d(inputs).
    """
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


@dataclass
class LossWeights:
    w_pde: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    w_data: float = 1.0


def pde_residual_theta_tau_minus_theta_xx(model: nn.Module, xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Residual for nondimensional 1D heat equation:
      theta_tau - theta_xi_xi = 0
    """
    # Ensure inputs require grad so derivatives can be computed
    xi = xi.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)

    # Concatenate spatial and temporal coordinates and evaluate network
    X = torch.cat([xi, tau], dim=1)
    theta = model(X)

    # Compute derivatives using autograd helpers
    theta_tau = _grad(theta, tau)
    theta_xi = _grad(theta, xi)
    theta_xixi = _grad(theta_xi, xi)

    # Residual R = theta_tau - theta_xi_xi
    return theta_tau - theta_xixi


def predict_theta(model: nn.Module, xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    X = torch.cat([xi, tau], dim=1)
    return model(X)


def dtheta_dxi(model: nn.Module, xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # Compute first derivative of theta with respect to xi
    xi = xi.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)
    theta = model(torch.cat([xi, tau], dim=1))
    theta_xi = _grad(theta, xi)
    return theta_xi


def compute_losses(
    model: nn.Module,
    batch,
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    batch is PINNBatch from src.data
    """
    # PDE residual loss (enforce PDE at collocation points)
    r = pde_residual_theta_tau_minus_theta_xx(model, batch.xi_r, batch.tau_r)
    loss_pde = torch.mean(r**2)

    # Initial condition loss (theta predicted vs. given at tau=0)
    theta_ic_hat = predict_theta(model, batch.xi_ic, batch.tau_ic)
    loss_ic = torch.mean((theta_ic_hat - batch.theta_ic) ** 2)

    # Right boundary condition (Neumann flux) loss
    flux_bc_hat = dtheta_dxi(model, batch.xi_bc, batch.tau_bc)
    loss_bc = torch.mean((flux_bc_hat - batch.flux_bc) ** 2)

    # Optional interior data loss (if available)
    loss_data = torch.tensor(0.0, device=loss_pde.device)
    if batch.xi_data is not None and batch.tau_data is not None and batch.theta_data is not None:
        theta_data_hat = predict_theta(model, batch.xi_data, batch.tau_data)
        loss_data = torch.mean((theta_data_hat - batch.theta_data) ** 2)

    # Weighted total loss used for training
    total = (
        weights.w_pde * loss_pde
        + weights.w_ic * loss_ic
        + weights.w_bc * loss_bc
        + weights.w_data * loss_data
    )

    logs = {
        "total": float(total.detach().cpu().item()),
        "pde": float(loss_pde.detach().cpu().item()),
        "ic": float(loss_ic.detach().cpu().item()),
        "bc": float(loss_bc.detach().cpu().item()),
        "data": float(loss_data.detach().cpu().item()),
    }
    return total, logs
