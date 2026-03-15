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
from typing import Dict, Optional, Tuple, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden: Sequence[int] | int = 30,
        layers: int = 2,
        out_dim: int = 1,
    ):
        """Flexible fully-connected MLP with Tanh activations.

        Two ways to specify the architecture:
        - `hidden` as an int and `layers` as number of layers (>=2):
            creates [in_dim -> hidden] + (layers-2) * [hidden -> hidden] + [hidden -> out_dim]
        - `hidden` as a sequence of ints: each entry specifies the number of neurons
            in that hidden layer, e.g. `hidden=[64, 64, 32]` builds
            in_dim->64->64->32->out_dim. When `hidden` is a sequence, `layers` is ignored.
        """
        super().__init__()

        # Build list of hidden layer sizes
        if isinstance(hidden, (list, tuple)):
            sizes = list(hidden)
            if len(sizes) < 1:
                raise ValueError("hidden sequence must contain at least one layer size")
        else:
            # integer hidden size + number of layers
            if layers < 2:
                raise ValueError("layers must be >= 2 when `hidden` is an int")
            # first hidden layer + (layers-2) intermediate hidden layers
            sizes = [int(hidden)] * (layers - 1)

        # Assemble nn.Sequential: in_dim -> sizes[0] -> ... -> sizes[-1] -> out_dim
        net: list[nn.Module] = []
        # first layer
        net.append(nn.Linear(in_dim, sizes[0]))
        net.append(nn.Tanh())

        # intermediate hidden layers
        for prev, curr in zip(sizes[:-1], sizes[1:]):
            net.append(nn.Linear(prev, curr))
            net.append(nn.Tanh())

        # final linear to output
        net.append(nn.Linear(sizes[-1], out_dim))
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


def _grad(outputs: torch.Tensor, inputs: torch.Tensor, create_graph: bool = True, retain_graph: bool = True) -> torch.Tensor:
    """Compute gradient of `outputs` w.r.t. `inputs` using PyTorch autograd.

    Returns tensor with same shape as `inputs` representing d(outputs)/d(inputs).

    Args:
        create_graph: whether to create the graph for higher-order derivatives.
        retain_graph: whether to retain the graph for further operations.
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


@dataclass
class FluxLearnedConfig:
    """Configuration for unknown nondimensional flux q_hat(tau)."""
    tau_knots: torch.Tensor
    q_ctrl: torch.nn.Parameter
    lambda_smooth: float = 0.0


def interp1d_linear(
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    """Simple linear interpolation with clamped ends.

    Inputs:
    - x_knots: shape (M,) increasing
    - y_knots: shape (M,) or (M,1)
    - x_query: shape (N,1) or (N,)
    Returns:
    - y_query: shape (N,1)
    """
    xk = x_knots.reshape(-1)
    yk = y_knots.reshape(-1)
    xq = x_query.reshape(-1)

    # bucketize -> indices in [0, M]
    idx = torch.bucketize(xq, xk)
    idx = idx.clamp(1, xk.numel() - 1)

    x0 = xk[idx - 1]
    x1 = xk[idx]
    y0 = yk[idx - 1]
    y1 = yk[idx]

    t = (xq - x0) / (x1 - x0 + 1e-12)
    y = y0 + t * (y1 - y0)
    return y.reshape(-1, 1)


def pde_residual_theta_tau_minus_theta_xi_xx(
    model: nn.Module,
    xi: torch.Tensor,
    tau: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Residual for nondimensional 1D heat equation:
      theta_tau - theta_xi_xi = 0

    Args:
        create_graph: whether to build graph for higher-order derivatives.
    """
    # Ensure inputs require grad so derivatives can be computed
    xi = xi.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)

    # Concatenate spatial and temporal coordinates and evaluate network
    X = _stack_input(xi, tau, mu)
    theta = model(X)

    # Compute derivatives using autograd helpers
    theta_tau = _grad(theta, tau, create_graph=create_graph, retain_graph=True)
    theta_xi = _grad(theta, xi, create_graph=create_graph, retain_graph=True)
    theta_xixi = _grad(theta_xi, xi, create_graph=create_graph, retain_graph=False)

    # Residual R = theta_tau - theta_xi_xi
    r = theta_tau - theta_xixi
    if r.shape != xi.shape:
        raise AssertionError(f"PDE residual shape mismatch: {r.shape} vs xi {xi.shape}")
    return r


def predict_theta(model: nn.Module, xi: torch.Tensor, tau: torch.Tensor, mu: Optional[torch.Tensor] = None) -> torch.Tensor:
    X = _stack_input(xi, tau, mu)
    return model(X)


def _stack_input(xi: torch.Tensor, tau: torch.Tensor, mu: Optional[torch.Tensor]) -> torch.Tensor:
    X = torch.cat([xi, tau], dim=1)
    if mu is not None:
        X = torch.cat([X, mu], dim=1)
    return X


def dtheta_dxi(
    model: nn.Module,
    xi: torch.Tensor,
    tau: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    create_graph: bool = True,
) -> torch.Tensor:
    # Compute first derivative of theta with respect to xi
    xi = xi.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)
    X = _stack_input(xi, tau, mu)
    theta = model(X)
    theta_xi = _grad(theta, xi, create_graph=create_graph, retain_graph=False)
    return theta_xi


def compute_losses(
    model: nn.Module,
    batch,
    weights: LossWeights,
    flux_mode: str = "known",
    flux_cfg: Optional[FluxLearnedConfig] = None,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    batch is PINNBatch from src.data

    Args:
        create_graph: whether to create autograd graph (set False for validation).
    """
    # Validate flux API.
    flux_mode = str(flux_mode).lower()
    if flux_mode not in {"known", "unknown"}:
        raise ValueError(f"Invalid flux_mode '{flux_mode}'. Expected 'known' or 'unknown'.")

    # Ensure we can create fallback tensors even if some inputs are empty.
    device = getattr(batch, "xi_r", torch.tensor(0)).device

    # Initial condition loss (theta predicted vs. given at tau=0)
    mu_ic = getattr(batch, "mu_ic", None)
    theta_ic_hat = predict_theta(model, batch.xi_ic, batch.tau_ic, mu_ic)
    loss_ic = torch.mean((theta_ic_hat - batch.theta_ic) ** 2) if batch.xi_ic.numel() > 0 else torch.tensor(0.0, device=device)

    # Optional interior data loss (if available)
    loss_data = torch.tensor(0.0, device=device)
    if batch.xi_data is not None and batch.tau_data is not None and batch.theta_data is not None:
        mu_data = getattr(batch, "mu_data", None)
        theta_data_hat = predict_theta(model, batch.xi_data, batch.tau_data, mu_data)
        loss_data = torch.mean((theta_data_hat - batch.theta_data) ** 2) if batch.xi_data.numel() > 0 else torch.tensor(0.0, device=device)

    # PDE residual loss (enforce PDE at collocation points)
    mu_r = getattr(batch, "mu_r", None)
    r = pde_residual_theta_tau_minus_theta_xi_xx(
        model, batch.xi_r, batch.tau_r, mu_r, create_graph=create_graph
    )
    loss_pde = torch.mean(r**2) if batch.xi_r.numel() > 0 else torch.tensor(0.0, device=device)

    # Left boundary condition loss (enforce theta at xi=0).
    # In the nondimensionalization used by the dataset, the left boundary is typically
    # fixed at 0 (theta = 0), but this can be overridden by providing `batch.theta_bc`.
    if weights.w_bc != 0.0 and batch.xi_bc.numel() > 0:
        mu_bc = getattr(batch, "mu_bc", None)
        theta_bc_target = getattr(batch, "theta_bc", None)
        theta_bc_hat = predict_theta(model, batch.xi_bc, batch.tau_bc, mu_bc)
        if theta_bc_target is None:
            theta_bc_target = torch.zeros_like(theta_bc_hat)
        loss_bc = torch.mean((theta_bc_hat - theta_bc_target) ** 2)
    else:
        loss_bc = torch.tensor(0.0, device=device)

    # Weighted total loss used for training
    total = (
        weights.w_pde * loss_pde
        + weights.w_ic * loss_ic
        + weights.w_bc * loss_bc
        + weights.w_data * loss_data
    )
    if not torch.isfinite(total):
        print(f"Total loss not finite: {total.item()}")
        print(f"loss_pde: {loss_pde.item()}")
        print(f"loss_ic: {loss_ic.item()}")
        print(f"loss_bc: {loss_bc.item()}")
        print(f"loss_data: {loss_data.item()}")
        raise AssertionError("Total loss is not finite.")

    logs = {
        "total": float(total.detach().cpu().item()),
        "pde": float(loss_pde.detach().cpu().item()),
        "ic": float(loss_ic.detach().cpu().item()),
        "bc": float(loss_bc.detach().cpu().item()),  # BC loss (not optimized when w_bc=0)
        "bc_monitor": 0.0,  # Disabled during training to avoid graph conflicts
        "data": float(loss_data.detach().cpu().item()),
        "smooth": 0.0,
    }
    return total, logs
