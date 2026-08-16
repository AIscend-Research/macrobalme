# =====================================================================================
# 1. Environment, configuration, reproducibility
# =====================================================================================
import os, sys, json, math, time, itertools, random, hashlib, textwrap, warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np

warnings.filterwarnings("ignore")

# ---- output directory (works on Kaggle and locally) ---------------------------------
OUT = "/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else "outputs"
for sub in ("figures", "images", "svg", "html", "tables", "data", "anim"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
P = lambda *a: os.path.join(OUT, *a)

# ---- global scale switch -------------------------------------------------------------
# QUICK=True   -> ~8-12 min end-to-end on a Kaggle CPU kernel (smoke test, all outputs produced)
# QUICK=False  -> ~1.5-3 h, the numbers reported in the paper
QUICK = bool(int(os.environ.get("MB_QUICK", "1")))


@dataclass
class Config:
    seed: int = 20260815
    grid: int = 48                 # GCell grid is grid x grid
    die: float = 1000.0            # die edge, um
    # --- oracle fidelity ---
    cap_k: float = 1.0             # track supply = cap_k * cap_pct-quantile of unconstrained demand
    cap_pct: float = 97.0
    place_iters: int = 90          # ePlace-lite gradient steps
    route_iters: int = 6           # negotiated-congestion rip-up & reroute rounds
    # --- experiment sizes (overridden by QUICK below) ---
    n_interventions: int = 420     # real oracle interventions per design (surrogate training set)
    shapley_perms: int = 4000      # Monte-Carlo permutations (surrogate-evaluated)
    resp_max_witness: int = 3      # max |W| searched for HP responsibility
    unet_epochs: int = 120
    gnn_epochs: int = 400
    verify_topk: int = 4           # attributions re-verified with the real oracle
    repair_topk: int = 3           # macros edited in the closed-loop repair experiment
    repair_candidates: int = 12    # legal sites tried per edited macro

    def quicken(self):
        self.grid = 40
        self.place_iters = 55
        self.route_iters = 4
        self.n_interventions = 130
        self.shapley_perms = 900
        self.resp_max_witness = 2
        self.unet_epochs = 35
        self.gnn_epochs = 150
        self.verify_topk = 3
        self.repair_topk = 2
        self.repair_candidates = 6
        return self


CFG = Config()
if QUICK:
    CFG.quicken()

random.seed(CFG.seed)
np.random.seed(CFG.seed % (2**31))

# ---- optional deps -------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(CFG.seed)
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_TORCH = True
except Exception as e:                                    # pragma: no cover
    HAS_TORCH, DEV = False, None
    print("torch unavailable -> surrogates fall back to numpy ridge models:", e)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except Exception:                                          # pragma: no cover
    HAS_PLOTLY = False

import pandas as pd

print(f"outputs -> {os.path.abspath(OUT)}")
print(f"QUICK={QUICK}  torch={HAS_TORCH} ({DEV})  plotly={HAS_PLOTLY}  grid={CFG.grid}")
