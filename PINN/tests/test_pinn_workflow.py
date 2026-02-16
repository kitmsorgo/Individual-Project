from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_manifest_rows, load_case_manifest_row, PINNBatch
from src.pinn import MLP, LossWeights, FluxLearnedConfig, pde_residual_theta_tau_minus_theta_xx, compute_losses


def _pinn_root() -> Path:
    here = Path.cwd()
    if (here / "PINN" / "src").exists():
        return here / "PINN"
    if (here / "src").exists():
        return here
    raise RuntimeError("Run tests from repo root or PINN root.")


def test_pde_residual_shape() -> None:
    model = MLP(hidden=8, layers=2)
    xi = torch.rand(32, 1, dtype=torch.float32)
    tau = torch.rand(32, 1, dtype=torch.float32)
    r = pde_residual_theta_tau_minus_theta_xx(model, xi, tau)
    assert r.shape == xi.shape


def test_tau_nondimensionalization() -> None:
    root = _pinn_root()
    row = load_manifest_rows(root / "data" / "manifest.csv")[0]
    case = load_case_manifest_row(row, root=root)
    t = case["raw"]["t"]
    alpha = case["physical"]["alpha"]
    L = case["physical"]["L"]
    tau_expected = alpha * t / (L**2)
    assert case["nondim"]["tau"].shape == tau_expected.shape
    assert torch.allclose(
        torch.tensor(case["nondim"]["tau"]),
        torch.tensor(tau_expected, dtype=torch.float32),
        rtol=1e-6,
        atol=1e-8,
    )


def test_bc_enforcement_known_and_unknown_modes() -> None:
    device = torch.device("cpu")
    model = MLP(hidden=8, layers=2).to(device)
    weights = LossWeights()
    n = 16
    zeros = torch.zeros(n, 1, device=device)
    rand = torch.rand(n, 1, device=device)

    known_batch = PINNBatch(
        xi_r=rand,
        tau_r=rand,
        xi_ic=rand,
        tau_ic=zeros,
        theta_ic=zeros,
        xi_bc=torch.ones_like(rand),
        tau_bc=rand,
        flux_bc=zeros,
        xi_data=rand,
        tau_data=rand,
        theta_data=zeros,
    )
    total_known, _ = compute_losses(model, known_batch, weights, flux_mode="known", flux_cfg=None)
    assert torch.isfinite(total_known)

    unknown_batch = PINNBatch(
        xi_r=rand,
        tau_r=rand,
        xi_ic=rand,
        tau_ic=zeros,
        theta_ic=zeros,
        xi_bc=torch.ones_like(rand),
        tau_bc=rand,
        flux_bc=None,
        xi_data=rand,
        tau_data=rand,
        theta_data=zeros,
    )
    tau_knots = torch.linspace(0.0, 1.0, 10, device=device)
    q_ctrl = torch.nn.Parameter(torch.zeros(10, device=device))
    flux_cfg = FluxLearnedConfig(tau_knots=tau_knots, q_ctrl=q_ctrl, lambda_smooth=1e-4)
    total_unknown, _ = compute_losses(model, unknown_batch, weights, flux_mode="unknown", flux_cfg=flux_cfg)
    assert torch.isfinite(total_unknown)
