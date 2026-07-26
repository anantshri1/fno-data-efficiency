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

def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x, mean, std):
    return x * std + mean

def relative_l2(pred: jax.Array, true: jax.Array) -> jax.Array:
    """
    Relative L2 error averaged over a batch.
    pred, true: (B, nx)
    """
    diff_norm = jnp.linalg.norm(pred-true, axis=-1)
    true_norm = jnp.linalg.norm(true, axis=-1)
    return jnp.mean(diff_norm / true_norm)

"""
Diagnostics:

```python
if __name__ == "__main__":
    cfg = Config()

    # load data
    u0_train, uT_train = load_dataset("data/train.npz")
    u0_test,  uT_test  = load_dataset("data/test.npz")
    print(f"train: {u0_train.shape}, test: {u0_test.shape}")

    # convert to jax arrays
    u0_train = jnp.array(u0_train)
    uT_train = jnp.array(uT_train)
    u0_test  = jnp.array(u0_test)
    uT_test  = jnp.array(uT_test)

    # relative L2 of a random (untrained) predictor — should be ~1.0
    key = jax.random.PRNGKey(0)
    random_pred = jax.random.normal(key, uT_test.shape)
    err = relative_l2(random_pred, uT_test)
    print(f"random predictor rel-L2: {err:.4f}")   # expect ~1.0
```
"""

"""
The training loop:
* Training in JAX: in PyTorch, `loss.backward()` mutates gradients in-place.
In JAX, `loss_fn(model, batch) -> scalar` and call `jax.value_and_grad` on it. 
This returns the loss and gradiet of the loss wrt the model. `optax` produces a new model.
* `eqx.filter_value_and_grad`: a thin wrapper around `jax.value_and_grad` that knows to only differentiate through array leaves of the model `pytree`, ignoring static fields like `n_modes`. 
If you used raw `jax.value_and_grad` you'd have to handle this split manually.
* `eqx.filter_jit`: same idea — `jit`s the function but correctly handles the static/dynamic split in Equinox modules. 
Always use this instead of raw `jax.jit` when the function takes an `eqx.Module` as argument.
"""

@eqx.filter_jit
def train_step(model, u0_batch, uT_batch, opt_state, optimizer):
    """
    One gradient; returns updated model, opt_state and loss.
    """
    def loss_fn(model):
        # vmap the model over the batch axis.
        # model(u0) handles one sample; vmap makes it handle (B, nx).
        pred = jax.vmap(model)(u0_batch)
        return relative_l2(pred, uT_batch)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, opt_state_new = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss

@eqx.filter_jit
def eval_step(model, u0_batch, uT_batch):
    pred = jax.vmap(model)(u0_batch)
    return relative_l2(pred, uT_batch)

"""
Diagnostics

```python
if __name__ == "__main__":
    cfg = Config()

    u0_train = jnp.array(np.load("data/train.npz")["u0"])
    uT_train = jnp.array(np.load("data/train.npz")["uT"])
    u0_test  = jnp.array(np.load("data/test.npz")["u0"])
    uT_test  = jnp.array(np.load("data/test.npz")["uT"])

    key = jax.random.PRNGKey(0)
    model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
                n_blocks=cfg.n_fno_blocks, key=key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # one step on a small batch
    u0_batch  = u0_train[:8]
    uT_batch  = uT_train[:8]

    model, opt_state, loss = train_step(model, u0_batch, uT_batch,
                                        opt_state, optimizer)
    print(f"loss after 1 step: {loss:.4f}")   # should be < 1.98, roughly ~1
    print("opt_state type:", type(opt_state))
```
"""

"""
Some JAX specific ponts:
* JAX has no `DataLoader`. You shuffle indices with `jax.random.permutation` (not `numpy.random` — 
inside jitted code everything must go through JAX's PRNG) then slice the array.
* JAX's random functions are pure — `jax.random.permutation(key, n)` always returns the same permutation for the same key. This makes experiments reproducible and jit-safe. 
* `eqx.tree_serialise_leaves` saves all array leaves of the model pytree to disk. The companion `eqx.tree_deserialise_leaves` loads them back into a fresh model instance.
"""

def make_batches(u0, uT, batch_size, key):
    """
    Shuffle and yield (u0_batch, uT_batch) pairs for one epoch.
    Drops the last incomplete batch.
    """
    n = u0.shape[0]
    perm = jax.random.permutation(key, n)   # JAX PRNG, not numpy
    u0, uT = u0[perm], uT[perm]
    n_batches = n // batch_size
    for i in range(n_batches):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        yield u0[sl], uT[sl]


def train(cfg, n_train: int = None):
    # --- data ---
    u0_tr = jnp.array(np.load("data/train.npz")["u0"])
    uT_tr = jnp.array(np.load("data/train.npz")["uT"])
    u0_te = jnp.array(np.load("data/test.npz")["u0"])
    uT_te = jnp.array(np.load("data/test.npz")["uT"])

    if n_train is not None:
        u0_tr, uT_tr = u0_tr[:n_train], uT_tr[:n_train]

    # normalize using training statistics only
    u0_mean, u0_std = u0_tr.mean(), u0_tr.std()
    uT_mean, uT_std = uT_tr.mean(), uT_tr.std()

    u0_tr = normalize(u0_tr, u0_mean, u0_std)
    uT_tr = normalize(uT_tr, uT_mean, uT_std)
    u0_te = normalize(u0_te, u0_mean, u0_std)
    uT_te = normalize(uT_te, uT_mean, uT_std)

    print(f"training on {u0_tr.shape[0]} samples")
    #print(f"u0 train | mean: {float(u0_tr.mean()):.4f}, std: {float(u0_tr.std()):.4f}")
    #print(f"uT train | mean: {float(uT_tr.mean()):.4f}, std: {float(uT_tr.std()):.4f}")
    #print(f"u0 test  | mean: {float(u0_te.mean()):.4f}, std: {float(u0_te.std()):.4f}")

    # --- model + optimiser with step LR decay ---
    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
                n_blocks=cfg.n_fno_blocks, key=mk)

    n_batches = u0_tr.shape[0] // cfg.batch_size
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
    mlflow.set_experiment("fno-data-efficiency")
    with mlflow.start_run(run_name=f"fno_N{n_train or cfg.n_train_max}"):
        mlflow.log_params({
            "n_train":    n_train or cfg.n_train_max,
            "n_modes":    cfg.n_modes,
            "n_channels": cfg.n_channels,
            "n_fno_blocks": cfg.n_fno_blocks,
            "lr":         cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "n_epochs":   cfg.n_epochs,
            "seed":       cfg.seed,
        })

        for epoch in range(cfg.n_epochs):
            key, sk = jax.random.split(key)

            # --- train ---
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
                print(f"epoch {epoch:3d} | "
                      f"train {train_loss:.4f} | test {test_loss:.4f}")

            # checkpoint every 50 epochs
            if epoch % 50 == 0:
                eqx.tree_serialise_leaves(
                    f"checkpoints/fno_epoch{epoch:04d}.eqx", model
                )

        # final checkpoint
        eqx.tree_serialise_leaves("checkpoints/fno_final.eqx", model)
        print(f"\ndone. final test rel-L2: {test_loss:.4f}")

    return model


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
