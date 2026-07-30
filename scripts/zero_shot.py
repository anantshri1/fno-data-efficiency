import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from scripts.train_model import build_model, normalize, relative_l2


def training_stats(n_train: int):
    """Recompute the exact normalization stats used during training.
    Global scalars over (n_train, nx=256), so they transfer to any resolution."""
    d = np.load("data/train.npz")
    u0_tr = jnp.array(d["u0"])[:n_train]
    uT_tr = jnp.array(d["uT"])[:n_train]
    return (u0_tr.mean(), u0_tr.std(), uT_tr.mean(), uT_tr.std())


def load_model(model_type: str, n_train: int, seed: int, cfg: Config) -> eqx.Module:
    """Rebuild the model skeleton with training hyperparams, then load weights into it.
    eqx.tree_deserialise_leaves needs a correctly-shaped pytree to fill — the
    PRNGKey here is irrelevant since every leaf gets overwritten."""
    path = f"checkpoints/{model_type}_N{n_train}_seed{seed}_final.eqx"
    skeleton = build_model(model_type, cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, skeleton)


def eval_at_resolution(model, u0, uT, stats) -> float:
    """Normalize inputs with training stats, run model, compare in normalized space
    (matching how the Phase 4 test numbers were computed)."""
    u0_mean, u0_std, uT_mean, uT_std = stats
    u0n = normalize(u0, u0_mean, u0_std)
    uTn = normalize(uT, uT_mean, uT_std)
    pred_n = jax.vmap(model)(u0n)
    return float(relative_l2(pred_n, uTn))



if __name__ == "__main__":
    cfg = Config()
    stats = training_stats(2000)

    d = np.load("data/test_superres.npz")
    u0_256 = jnp.array(d["u0_256"])
    uT_256 = jnp.array(d["uT_256"])
    print(f"superres test set @ 256: {u0_256.shape}")

    fno = load_model("fno", 2000, 0, cfg)
    err = eval_at_resolution(fno, u0_256, uT_256, stats)
    print(f"FNO @ nx=256: rel-L2 = {err:.4f}   (expect ~0.02-0.03)")

    # --- extend: all resolutions, both models ---
    resolutions = [256, 512, 1024]
    print("\n" + "=" * 44)
    print(f"{'model':>6} | {'nx':>5} | {'rel-L2':>8}")
    print("-" * 44)

    for model_type in ["fno", "unet"]:
        model = load_model(model_type, 2000, 0, cfg)
        for nx in resolutions:
            u0 = jnp.array(d[f"u0_{nx}"])
            uT = jnp.array(d[f"uT_{nx}"])
            err = eval_at_resolution(model, u0, uT, stats)
            print(f"{model_type:>6} | {nx:>5} | {err:>8.4f}")
        print("-" * 44)

    # --- DIAGNOSTIC: is FNO actually evaluating at different resolutions? ---
    print("\n--- FNO resolution diagnostic ---")
    fno = load_model("fno", 2000, 0, cfg)
    for nx in [256, 512, 1024]:
        u0 = jnp.array(d[f"u0_{nx}"])
        pred = jax.vmap(fno)(normalize(u0, stats[0], stats[1]))
        print(f"nx={nx:>4} | u0 shape {u0.shape} | pred shape {pred.shape} "
              f"| pred mean {float(pred.mean()):+.5f} | pred std {float(pred.std()):.5f}")

    # are the coarser grids exact subsamples of the finest?
    u0_1024 = jnp.array(d["u0_1024"])
    u0_512  = jnp.array(d["u0_512"])
    u0_256  = jnp.array(d["u0_256"])
    print("u0_512 == u0_1024[:, ::2]?", bool(jnp.allclose(u0_512, u0_1024[:, ::2])))
    print("u0_256 == u0_1024[:, ::4]?", bool(jnp.allclose(u0_256, u0_1024[:, ::4])))