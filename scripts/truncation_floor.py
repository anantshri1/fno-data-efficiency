import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp

from configs.default import Config


def relative_l2(pred, true):
    diff_norm = jnp.linalg.norm(pred - true, axis=-1)
    true_norm = jnp.linalg.norm(true, axis=-1)
    return jnp.mean(diff_norm / true_norm)


def truncation_floor(uT: jnp.ndarray, n_modes: int) -> float:
    """
    Best-case error from representing uT with only the first n_modes
    Fourier modes. This is NOT a model — it's rfft -> zero high modes -> irfft,
    applied directly to ground truth. Establishes a ceiling on what any
    n_modes-truncated spectral representation can achieve.
    """
    nx = uT.shape[-1]

    uT_ft = jnp.fft.rfft(uT, axis=-1)          # (B, nx//2 + 1), complex64
    uT_ft_trunc = uT_ft.at[:, n_modes:].set(0.0)  # zero everything above n_modes
    uT_proj = jnp.fft.irfft(uT_ft_trunc, n=nx, axis=-1)

    return float(relative_l2(uT_proj, uT))


if __name__ == "__main__":
    cfg = Config()

    u0_te = jnp.array(np.load("data/test.npz")["u0"])
    u0_floor = truncation_floor(u0_te, cfg.n_modes)
    uT_te = jnp.array(np.load("data/test.npz")["uT"])
    print(f"test set: {uT_te.shape}")

    floor = truncation_floor(uT_te, cfg.n_modes)
    print(f"n_modes = {cfg.n_modes}")
    print(f"truncation floor (rel-L2): {floor:.4f}")
    print(f"u0 truncation floor (rel-L2): {u0_floor:.4f}")
    print(f"high-freq content added by Burgers' evolution: {floor - u0_floor:+.4f}")