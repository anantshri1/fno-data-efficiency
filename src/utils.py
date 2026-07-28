import jax
import equinox as eqx

def count_params(model: eqx.Module) -> int:
    """
    Count the total number of scalar array elements in an Equinox model.

    eqx.filter(model, eqx.is_array) strips out all non-array leaves
    (static ints like n_modes, strings, etc.) and returns a pytree
    containing only the JAX arrays.

    jax.tree_util.tree_leaves then flattens that pytree into a list,
    and we sum .size over each array leaf.

    Note: complex64 arrays count their *total elements* (real + imag
    packed together in memory), so a (16, 32, 32) complex64 array
    contributes 16*32*32 = 16,384 — not 32,768. Our FNO stores
    w_real and w_imag as *separate* float32 arrays, so they're
    counted correctly as two distinct leaves.
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(leaf.size for leaf in leaves)

"""
# Diagnostics
# quick inline test (paste in a python3 shell from fno-data-efficiency/)

```python
import sys; sys.path.insert(0, ".")
import jax
import equinox as eqx
from configs.default import Config
from src.fno import FNO
from src.utils import count_params

cfg = Config()
key = jax.random.PRNGKey(0)
model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes, n_blocks=cfg.n_fno_blocks, key=key)
print(count_params(model))   # expect 139,745
```
"""