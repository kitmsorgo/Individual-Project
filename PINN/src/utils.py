from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class Scales:
    # Physical scales used only for flux conversion (optional)
    k: float = 1.0          # W/mK
    L: float = 1.0          # m
    dT: float = 1.0         # K (Delta T used in theta scaling)

    @property
    def flux_scale(self) -> float:
        # q'' = -k*(dT/L)*dtheta/dxi
        return self.k * (self.dT / self.L)
