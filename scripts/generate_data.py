import sys
sys.path.insert(0, ".")

import jax
from configs.default import Config
from src.data_gen import generate_dataset, save_dataset

cfg = Config()
key = jax.random.PRNGKey(cfg.seed)

# --- Test set (fixed, never changes across experiments) ---
key, subkey = jax.random.split(key)
print(f"Generating test set ({cfg.n_test} samples)...")
u0_test, uT_test = generate_dataset(subkey, n_samples=cfg.n_test, cfg=cfg)
save_dataset(u0_test, uT_test, f"{cfg.data_dir}/test.npz")

# --- Training pool (generate n_train_max; subsets drawn at sweep time) ---
key, subkey = jax.random.split(key)
print(f"Generating training pool ({cfg.n_train_max} samples)...")
u0_train, uT_train = generate_dataset(subkey, n_samples=cfg.n_train_max, cfg=cfg)
save_dataset(u0_train, uT_train, f"{cfg.data_dir}/train.npz")

print("Done.")