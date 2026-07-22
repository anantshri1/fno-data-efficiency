import sys
sys.path.insert(0, ".")

import jax
import matplotlib.pyplot as plt
from configs.default import Config
from src.data_gen import sample_grf
from src.data_gen import solve_burgers
from src.data_gen import generate_dataset
import jax.numpy as jnp
import numpy as np


cfg = Config()
key = jax.random.PRNGKey(cfg.seed)

u0 = sample_grf(key, nx = cfg.nx, length_scale=cfg.grf_length_scale, n_samples=5)
print("Shape:", u0.shape)       # expect (5,256)
print("Mean:", u0.mean())         # expect ~0
print("Std:", u0.std())           # expect ~1

x = jax.numpy.linspace(0,1,cfg.nx)
plt.figure(figsize=(8,3))
for i in range(5):
    plt.plot(x, u0[i], alpha=0.7)
plt.title("5 GRF samples (initial conditions)")
plt.xlabel("x")
plt.tight_layout()
plt.show()

u0_single = u0[0]   # shape (256,)
u_T = solve_burgers(u0_single, nu=cfg.pde_nu, t_end=cfg.t_end, nt=cfg.nt)

print("Solver output shape:", u_T.shape)   # expect (256,)
print("u_T min/max:", u_T.min(), u_T.max())

x = jnp.linspace(0, 1, cfg.nx)
plt.figure(figsize=(8, 3))
plt.plot(x, u0_single, label="u(x,0)  initial")
plt.plot(x, u_T,       label="u(x,T)  evolved")
plt.legend()
plt.title("Burgers' solver: one trajectory")
plt.xlabel("x")
plt.tight_layout()
plt.show()

key, subkey = jax.random.split(key)
u0_batch, uT_batch = generate_dataset(subkey, n_samples = 10, cfg = cfg)

print("u0 shape:", u0_batch.shape)       # expect (10,256)
print("uT shape:", uT_batch.shape)       # expect (10,256)

# Relative L2 error - unit test for diagnostics
diff = jnp.linalg.norm(uT_batch - u0_batch, axis=-1)
norm = jnp.linalg.norm(u0_batch, axis=-1)
print("Mean relative change IC→T:", (diff/norm).mean())