import sys
sys.path.insert(0, ".")

import jax
from configs.default import Config
from src.data_gen import (
    generate_dataset,
    generate_dataset_superres,
    save_dataset,
    save_dataset_superres,
)

cfg = Config()
key = jax.random.PRNGKey(cfg.seed)

# --- Training pool (at base resolution nx=256) ---
key, subkey = jax.random.split(key)
print(f"Generating training pool ({cfg.n_train_max} samples at nx={cfg.nx})...")
u0_train, uT_train = generate_dataset(subkey, n_samples=cfg.n_train_max, cfg=cfg)
save_dataset(u0_train, uT_train, f"{cfg.data_dir}/train.npz")

# --- Test set (base resolution nx=256) ---
key, subkey = jax.random.split(key)
print(f"Generating test set ({cfg.n_test} samples at nx={cfg.nx})...")
u0_test, uT_test = generate_dataset(subkey, n_samples=cfg.n_test, cfg=cfg)
save_dataset(u0_test, uT_test, f"{cfg.data_dir}/test.npz")

# --- Super-resolution test set (solve at 1024, subsample to 512 and 256) ---
# NOTE: uses the same key as test set so ICs are independent from training pool
# but this is a separate set — do not reuse test.npz ICs here, 
# super-res has its own n_test samples solved at finest resolution
key, subkey = jax.random.split(key)
print(f"Generating super-res test set ({cfg.n_test} samples, "
      f"nx={cfg.nx}/{cfg.nx_mid}/{cfg.nx_super})...")
base, mid, sup = generate_dataset_superres(subkey, n_samples=cfg.n_test, cfg=cfg)
save_dataset_superres(base, mid, sup, f"{cfg.data_dir}/test_superres.npz")

print("All done.")