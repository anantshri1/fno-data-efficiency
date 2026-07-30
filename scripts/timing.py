import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.data_gen import solve_burgers
from scripts.train_model import normalize,build_model


def time_solver(u0_np: np.ndarray, cfg: Config, n_reps: int = 20) -> float:
    """Mean wall-clock seconds per call to solve_burgers, single sample."""
    # warmup (numpy solver, no compilation, but let filesystem/cache settle)
    _ = solve_burgers(u0_np, nu=cfg.pde_nu, t_end=cfg.t_end, nt=cfg.nt)

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = solve_burgers(u0_np, nu=cfg.pde_nu, t_end=cfg.t_end, nt=cfg.nt)
    t1 = time.perf_counter()
    return (t1 - t0) / n_reps


def time_fno(model, u0n: jnp.ndarray, n_reps: int = 20) -> float:
    """Mean wall-clock seconds per call to FNO inference, single sample.
    u0n: already-normalized single sample, shape (nx,)."""
    fwd = eqx.filter_jit(model)

    # warmup: triggers compilation, block_until_ready ensures it's finished
    out = fwd(u0n)
    jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(n_reps):
        out = fwd(u0n)
        jax.block_until_ready(out)   # force sync; async dispatch would undercount
    t1 = time.perf_counter()
    return (t1 - t0) / n_reps


if __name__ == "__main__":
    cfg = Config()

    d = np.load("data/test.npz")
    u0_sample_np = d["u0"][0]                       # single sample, raw (unnormalized)

    u0_tr = jnp.array(np.load("data/train.npz")["u0"])[:2000]
    u0_mean, u0_std = u0_tr.mean(), u0_tr.std()
    u0_sample_n = normalize(jnp.array(u0_sample_np), u0_mean, u0_std)

    solver_time = time_solver(u0_sample_np, cfg)
    print(f"FD solver:  {solver_time*1000:.3f} ms/sample")

    fno = build_model("fno", cfg, jax.random.PRNGKey(0))
    fno = eqx.tree_deserialise_leaves("checkpoints/fno_N2000_seed0_final.eqx", fno)
    fno_time = time_fno(fno, u0_sample_n)
    print(f"FNO:        {fno_time*1000:.3f} ms/sample")
    print(f"speedup:    {solver_time/fno_time:.1f}x")