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
from typing import Dict, Optional, Sequence, Tuple

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
        """Flexible fully-connected MLP with Tanh activations."""
        super().__init__()

        if isinstance(hidden, (list, tuple)):
            sizes = list(hidden)
            if len(sizes) < 1:
                raise ValueError("hidden sequence must contain at least one layer size")
        else:
            if layers < 2:
                raise ValueError("layers must be >= 2 when `hidden` is an int")
            sizes = [int(hidden)] * (layers - 1)

        net: list[nn.Module] = [nn.Linear(in_dim, sizes[0]), nn.Tanh()]
        for prev, curr in zip(sizes[:-1], sizes[1:]):
            net.append(nn.Linear(prev, curr))
            net.append(nn.Tanh())
        net.append(nn.Linear(sizes[-1], out_dim))
        self.net = nn.Sequential(*net)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=retain_graph,
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
    """Simple linear interpolation with clamped ends."""
    xk = x_knots.reshape(-1)
    yk = y_knots.reshape(-1)
    xq = x_query.reshape(-1)

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
    """Residual for the nondimensional 1D heat equation: theta_tau - theta_xi_xi = 0."""
    xi = xi.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)

    theta = model(_stack_input(xi, tau, mu))
    theta_tau = _grad(theta, tau, create_graph=create_graph, retain_graph=True)
    theta_xi = _grad(theta, xi, create_graph=True, retain_graph=True)
    theta_xixi = _grad(theta_xi, xi, create_graph=create_graph, retain_graph=create_graph)

    r = theta_tau - theta_xixi
    if r.shape != xi.shape:
        raise AssertionError(f"PDE residual shape mismatch: {r.shape} vs xi {xi.shape}")
    return r


def predict_theta(
    model: nn.Module,
    xi: torch.Tensor,
    tau: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return model(_stack_input(xi, tau, mu))


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
    xi = xi.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)
    theta = model(_stack_input(xi, tau, mu))
    return _grad(theta, xi, create_graph=create_graph, retain_graph=create_graph)


def compute_losses(
    model: nn.Module,
    batch,
    weights: LossWeights,
    flux_mode: str = "known",
    flux_cfg: FluxLearnedConfig | None = None,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PINN losses for parametric and legacy batch structures."""

    device = getattr(batch, "xi_r", torch.tensor(0.0)).device

    mu_ic = getattr(batch, "mu_ic", None)
    theta_ic_hat = predict_theta(model, batch.xi_ic, batch.tau_ic, mu_ic)
    loss_ic = (
        torch.mean((theta_ic_hat - batch.theta_ic) ** 2)
        if batch.xi_ic.numel() > 0
        else torch.tensor(0.0, device=device)
    )

    loss_data = torch.tensor(0.0, device=device)
    if batch.xi_data is not None and batch.tau_data is not None and batch.theta_data is not None:
        mu_data = getattr(batch, "mu_data", None)
        theta_data_hat = predict_theta(model, batch.xi_data, batch.tau_data, mu_data)
        loss_data = (
            torch.mean((theta_data_hat - batch.theta_data) ** 2)
            if batch.xi_data.numel() > 0
            else torch.tensor(0.0, device=device)
        )

    mu_r = getattr(batch, "mu_r", None)
    r = pde_residual_theta_tau_minus_theta_xi_xx(
        model,
        batch.xi_r,
        batch.tau_r,
        mu_r,
        create_graph=create_graph,
    )
    loss_pde = torch.mean(r**2) if batch.xi_r.numel() > 0 else torch.tensor(0.0, device=device)

    # Default parametric path:
    # - temperature targets in theta_bc enforce both left fixed Dirichlet and
    #   right-boundary temperature consistency with masking via NaNs.
    # Compatibility path:
    # - explicit Neumann/flux targets are still supported for legacy workflows.
    loss_bc = torch.tensor(0.0, device=device)
    if weights.w_bc != 0.0 and batch.xi_bc.numel() > 0:
        mu_bc = getattr(batch, "mu_bc", None)

        loss_temp_bc = torch.tensor(0.0, device=device)
        theta_bc_target = getattr(batch, "theta_bc", None)
        if theta_bc_target is not None and theta_bc_target.numel() > 0:
            theta_bc_hat = predict_theta(model, batch.xi_bc, batch.tau_bc, mu_bc)
            mask = ~torch.isnan(theta_bc_target)
            if mask.any():
                diff = theta_bc_hat[mask] - theta_bc_target[mask]
                loss_temp_bc = torch.mean(diff**2)

        loss_flux_bc = torch.tensor(0.0, device=device)
        flux_bc_target = getattr(batch, "flux_bc", None)
        if str(flux_mode).lower() == "unknown":
            if flux_cfg is None:
                raise ValueError("flux_mode='unknown' requires flux_cfg.")
            flux_bc_target = interp1d_linear(flux_cfg.tau_knots, flux_cfg.q_ctrl, batch.tau_bc)
        if flux_bc_target is not None and flux_bc_target.numel() > 0:
            theta_xi_bc = dtheta_dxi(
                model,
                batch.xi_bc,
                batch.tau_bc,
                mu_bc,
                create_graph=create_graph,
            )
            mask = ~torch.isnan(flux_bc_target)
            if mask.any():
                diff = theta_xi_bc[mask] - flux_bc_target[mask]
                loss_flux_bc = torch.mean(diff**2)

        loss_bc = loss_temp_bc + loss_flux_bc

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
        "bc": float(loss_bc.detach().cpu().item()),
        "bc_monitor": 0.0,
        "data": float(loss_data.detach().cpu().item()),
        "smooth": 0.0,
    }
    return total, logs
