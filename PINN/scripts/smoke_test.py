from __future__ import annotations

from pathlib import Path
import sys

import torch


def _repo_root() -> Path:
    here = Path.cwd()
    if (here / "PINN" / "src").exists():
        return here / "PINN"
    if (here / "src").exists():
        return here
    raise RuntimeError("Run smoke test from repo root or PINN root.")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_manifest_rows, load_case_manifest_row, build_train_val_batches_from_case
from src.pinn import MLP, LossWeights, FluxLearnedConfig
from src.train import train_adam, compute_losses_eval


def run_smoke(flux_mode: str = "known") -> None:
    flux_mode = flux_mode.lower()
    if flux_mode not in {"known", "unknown"}:
        raise ValueError("flux_mode must be 'known' or 'unknown'")

    root = ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = root / "data" / "manifest.csv"
    row = load_manifest_rows(manifest)[0]
    case = load_case_manifest_row(row, root=root)
    train_batch, val_batch = build_train_val_batches_from_case(
        case=case,
        device=device,
        flux_mode=flux_mode,
        n_r=512,
        val_fraction=0.2,
        seed=123,
    )

    model = MLP(hidden=16, layers=2).to(device)
    weights = LossWeights(w_pde=1.0, w_ic=1.0, w_bc=1.0, w_data=1.0)
    flux_cfg = None
    if flux_mode == "unknown":
        tau_end = float(case["nondim"]["tau"].max())
        tau_knots = torch.linspace(0.0, tau_end, 12, device=device)
        q_ctrl = torch.nn.Parameter(torch.zeros(12, device=device))
        flux_cfg = FluxLearnedConfig(tau_knots=tau_knots, q_ctrl=q_ctrl, lambda_smooth=1e-4)

    run_dir = root / "models" / "checkpoints" / "smoke"
    train_adam(
        model,
        train_batch,
        weights,
        flux_mode=flux_mode,
        flux_cfg=flux_cfg,
        lr=1e-3,
        max_steps=5,
        min_steps=1,
        eval_every=1,
        patience_evals=2,
        plateau_window=2,
        run_dir=run_dir,
        val_batch=val_batch,
        print_every=1,
    )
    _, logs = compute_losses_eval(model, val_batch, weights, flux_mode=flux_mode, flux_cfg=flux_cfg)
    print(f"[SMOKE][{flux_mode}] total={logs['total']:.4e} pde={logs['pde']:.4e} ic={logs['ic']:.4e} bc={logs['bc']:.4e} data={logs['data']:.4e}")


if __name__ == "__main__":
    run_smoke("known")
    run_smoke("unknown")
