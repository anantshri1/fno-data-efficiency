import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp

from configs.default import Config
from scripts.train_model import build_model, normalize
from scripts.zero_shot import load_model, training_stats  # reuse Phase-5 helpers


def banded_error_energy(model, u0, uT, stats, cutoff):
    """Fraction of signal energy the model fails to capture, per band.
    Sum error energy and signal energy over (samples x modes in band) FIRST,
    then divide once -- no per-mode division, so spectral nulls don't blow up.

    Returns (low_frac, high_frac): sqrt(sum|err|^2 / sum|true|^2) over each band."""
    u0_mean, u0_std, uT_mean, uT_std = stats
    u0n = normalize(u0, u0_mean, u0_std)
    uTn = normalize(uT, uT_mean, uT_std)
    pred_n = jax.vmap(model)(u0n)

    err_ft  = jnp.fft.rfft(pred_n - uTn, axis=-1)   # (B, K)
    true_ft = jnp.fft.rfft(uTn,          axis=-1)   # (B, K)

    err_e  = jnp.abs(err_ft)  ** 2                  # energy per (sample, mode)
    true_e = jnp.abs(true_ft) ** 2

    def band_frac(lo, hi):
        num = err_e[:, lo:hi].sum()                 # total error energy in band
        den = true_e[:, lo:hi].sum()                # total signal energy in band
        return float(jnp.sqrt(num / den))

    return band_frac(0, cutoff), band_frac(cutoff, err_ft.shape[-1])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def per_mode_energy_curve(model, u0, uT, stats):
    """Mean error energy per mode (sqrt of sample-averaged |err(k)|^2),
    for plotting the full spectrum. Not normalized per-mode -- we plot the
    raw error-energy shape and mark the cutoff."""
    u0_mean, u0_std, uT_mean, uT_std = stats
    u0n = normalize(u0, u0_mean, u0_std)
    uTn = normalize(uT, uT_mean, uT_std)
    pred_n = jax.vmap(model)(u0n)
    err_ft = jnp.fft.rfft(pred_n - uTn, axis=-1)
    return jnp.sqrt((jnp.abs(err_ft) ** 2).mean(axis=0))   # (K,)



if __name__ == "__main__":
    cfg = Config()
    stats = training_stats(2000)
    d = np.load("data/test.npz")
    u0, uT = jnp.array(d["u0"]), jnp.array(d["uT"])
    cutoff = cfg.n_modes

    print(f"{'model':>6} | {'low-band (k<16)':>16} | {'high-band (k>=16)':>18} | {'ratio':>6}")
    print("-" * 56)
    for model_type in ["fno", "unet"]:
        model = load_model(model_type, 2000, 0, cfg)
        low, high = banded_error_energy(model, u0, uT, stats, cutoff)
        print(f"{model_type:>6} | {low:>16.4f} | {high:>18.4f} | {high/low:>6.2f}")

    curves = {}
    for model_type in ["fno", "unet"]:
        model = load_model(model_type, 2000, 0, cfg)
        curves[model_type] = np.asarray(per_mode_energy_curve(model, u0, uT, stats))

    k = np.arange(curves["fno"].shape[0])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(k, curves["fno"],  label="FNO",    linewidth=1.8)
    ax.semilogy(k, curves["unet"], label="U-Net",  linewidth=1.8)
    ax.axvline(cutoff, color="k", linestyle="--", alpha=0.5,
               label=f"FNO cutoff (k={cutoff})")
    ax.set_xlabel("Fourier mode k")
    ax.set_ylabel("mean error magnitude  |pred - true|(k)")
    ax.set_title("Prediction error spectrum (N=2000, seed 0, nx=256)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/error_spectrum.png", dpi=150)
    print("saved results/error_spectrum.png")
