import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import mlflow

from configs.default import Config
from src.data_gen import load_dataset
from src.fno import FNO
from src.unet import UNet
from src.utils import count_params

def normalize(x, mean, std):
    return (x-mean)/std

def relative_l2(pred, true):
    diff_norm = jnp.linalg.norm(pred-true, axis=-1)
    true_norm = jnp.linalg.norm(true, axis=-1)
    return jnp.mean(diff_norm/true_norm)

def make_batches(u0, uT, batch_size, key):
    n = u0.shape[0]
    perm = jax.random.permutation(key,n)
    u0, uT = u0[perm], uT[perm]

    for i in range (n//batch_size):
        sl = slice(i* batch_size, (i+1)*batch_size)
        yield u0[sl], uT[sl]

@eqx.filter_jit
def train_step(model, u0_batch, uT_batch, opt_state, optimizer):
    def loss_fn(model):
        pred = jax.vmap(model)(u0_batch)
        return relative_l2(pred, uT_batch)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array)
                                              )
    return eqx.apply_updates(model, updates), opt_state_new, loss

@eqx.filter_jit
def eval_step(model, u0, uT):
    pred = jax.vmap(model)(u0)
    return relative_l2(pred, uT)

def build_model(model_type: str, cfg: Config, key: jax.Array) -> eqx.Module:
    if model_type == "fno":
        return FNO(d_v = cfg.n_channels, n_modes = cfg.n_modes,
                   n_blocks = cfg.n_fno_blocks, key = key)
    elif model_type == "unet":
        return UNet(base_channels= cfg.unet_base_channels, key = key
                    )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

# --- training loop ---------------------
def train(cfg: Config, model_type: str, n_train: int = None) -> eqx.Module:
    # --- data -----
    u0_tr = jnp.array(np.load("data/train.npz")["u0"])
    uT_tr = jnp.array(np.load("data/train.npz")["uT"])
    u0_te = jnp.array(np.load("data/test.npz")["u0"])
    uT_te = jnp.array(np.load("data/test.npz")["uT"])

    if n_train is not None:
        u0_tr, uT_tr = u0_tr[:n_train], uT_tr[:n_train]

    u0_mean, u0_std = u0_tr.mean(), u0_tr.std()
    uT_mean, uT_std = uT_tr.mean(), uT_tr.std()
    u0_tr = normalize(u0_tr, u0_mean, u0_std)
    uT_tr = normalize(uT_tr, uT_mean, uT_std)
    u0_te = normalize(u0_te, u0_mean, u0_std)
    uT_te = normalize(uT_te, uT_mean, uT_std)

    n_actual = u0_tr.shape[0]
    print(f"[{model_type}] training on {n_actual} samples (seed={cfg.seed})")

    # --- model + optimiser ---
    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = build_model(model_type, cfg, mk)
    print(f"[{model_type}] param count: {count_params(model):,}")

    n_batches = n_actual // cfg.batch_size
    schedule = optax.exponential_decay(
        init_value=cfg.learning_rate,
        transition_steps=cfg.lr_decay_every * n_batches,
        decay_rate=cfg.lr_decay_rate,
        staircase=True,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    os.makedirs("checkpoints", exist_ok=True)

    # --- MLflow ---
    mlflow.set_experiment(cfg.experiment_name)
    run_name = f"{model_type}_N{n_actual}_seed{cfg.seed}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type":   model_type,
            "n_train":      n_actual,
            "n_params":     count_params(model),
            "seed":         cfg.seed,
            "n_modes":      cfg.n_modes,      # only meaningful for FNO
            "n_channels":   cfg.n_channels,
            "n_fno_blocks": cfg.n_fno_blocks,
            "unet_base_channels": cfg.unet_base_channels,
            "lr":           cfg.learning_rate,
            "batch_size":   cfg.batch_size,
            "n_epochs":     cfg.n_epochs,
        })

        for epoch in range(cfg.n_epochs):
            key, sk = jax.random.split(key)

            batch_losses = []
            for u0b, uTb in make_batches(u0_tr, uT_tr, cfg.batch_size, sk):
                model, opt_state, loss = train_step(
                    model, u0b, uTb, opt_state, optimizer
                )
                batch_losses.append(float(loss))

            train_loss = float(np.mean(batch_losses))
            test_loss  = float(eval_step(model, u0_te, uT_te))

            mlflow.log_metrics(
                {"train_rel_l2": train_loss, "test_rel_l2": test_loss},
                step=epoch
            )

            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d} | train {train_loss:.4f} | test {test_loss:.4f}")

            if epoch % 50 == 0:
                ckpt = f"checkpoints/{model_type}_N{n_actual}_seed{cfg.seed}_epoch{epoch:04d}.eqx"
                eqx.tree_serialise_leaves(ckpt, model)

        ckpt_final = f"checkpoints/{model_type}_N{n_actual}_seed{cfg.seed}_final.eqx"
        eqx.tree_serialise_leaves(ckpt_final, model)
        print(f"  done. final test rel-L2: {test_loss:.4f}")
        print(f"  checkpoint: {ckpt_final}")

    return model

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fno", "unet"], required=True)
    parser.add_argument("--n_train", type=int, default=None,
                        help="Training set size. Defaults to cfg.n_train_max (2000).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. Overrides cfg.seed if provided.")
    args = parser.parse_args()

    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed

    train(cfg, model_type=args.model, n_train=args.n_train)
