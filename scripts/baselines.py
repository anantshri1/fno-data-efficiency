import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp

from configs.default import Config
from scripts.train_model import normalize, relative_l2


def identity_baseline(u0, uT):
    return float(relative_l2(u0, uT))


def linear_baseline(u0_tr, uT_tr, u0_te, uT_te):
    """Fit uT ~= A @ u0 (global linear operator) via least-squares on training data.
    u0_tr, uT_tr: (N, nx). Solves for A: (nx, nx) minimizing ||uT_tr - u0_tr @ A.T||^2,
    i.e. per-sample: uT ~= A @ u0. Using lstsq on u0_tr (N, nx) -> uT_tr (N, nx)
    directly solves for A.T of shape (nx, nx)."""
    # lstsq(a, b) solves a @ x = b in least-squares sense.
    # We want A such that u0 @ A.T ~= uT, i.e. (u0_tr) @ (A.T) ~= uT_tr.
    A_T, residuals, rank, sv = jnp.linalg.lstsq(u0_tr, uT_tr, rcond=None)
    uT_pred = u0_te @ A_T
    return float(relative_l2(uT_pred, uT_te)), A_T


if __name__ == "__main__":
    cfg = Config()

    u0_tr = jnp.array(np.load("data/train.npz")["u0"])[:2000]
    uT_tr = jnp.array(np.load("data/train.npz")["uT"])[:2000]
    u0_te = jnp.array(np.load("data/test.npz")["u0"])
    uT_te = jnp.array(np.load("data/test.npz")["uT"])

    # normalize with training stats, same convention as every other Phase 4/5 number
    u0_mean, u0_std = u0_tr.mean(), u0_tr.std()
    uT_mean, uT_std = uT_tr.mean(), uT_tr.std()
    u0_tr_n = normalize(u0_tr, u0_mean, u0_std)
    uT_tr_n = normalize(uT_tr, uT_mean, uT_std)
    u0_te_n = normalize(u0_te, u0_mean, u0_std)
    uT_te_n = normalize(uT_te, uT_mean, uT_std)

    id_err = identity_baseline(u0_te_n, uT_te_n)
    lin_err, _ = linear_baseline(u0_tr_n, uT_tr_n, u0_te_n, uT_te_n)

    print(f"identity baseline (uT = u0):        rel-L2 = {id_err:.4f}")
    print(f"linear operator (least-squares fit): rel-L2 = {lin_err:.4f}")
    print(f"--- for reference ---")
    print(f"FNO (N=2000):   0.0202-0.0230")
    print(f"U-Net (N=2000): 0.0656-0.0681")