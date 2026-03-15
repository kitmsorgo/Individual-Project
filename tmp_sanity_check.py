import sys
import torch
import numpy as np

sys.path.insert(0, r'C:/Users/wscm13/OneDrive - Loughborough University/Part C/IDP/Individual Project/PINN')

from src.pinn import MLP, LossWeights, compute_losses
from src.data import build_synthetic_batch

# synthetic test: zero field and zero flux
batch = build_synthetic_batch(
    n_r=10,
    n_ic=10,
    n_bc=10,
    tau_max=1.0,
    device=torch.device('cpu'),
    ic_fn=lambda xi: np.zeros_like(xi),
    bc_right_flux_fn=lambda tau: np.zeros_like(tau),
    n_data=10,
    data_fn=lambda xi, tau: np.zeros_like(xi),
)

model = MLP(in_dim=2, hidden=16, layers=3)
weights = LossWeights(w_pde=1.0, w_ic=1.0, w_bc=1.0, w_data=1.0)

loss, logs = compute_losses(model, batch, weights, flux_mode='known', create_graph=False)
print('loss', loss.item())
print('logs', logs)
