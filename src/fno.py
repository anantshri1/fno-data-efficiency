"""
What this block contains:
* eqx.Module: Equinox's base class for neural network layers. It automatically registers the class as a JAX pytree, meaning JAX's jit, grad, vmap etc. can see inside it and treat its array fields as leaves (differentiable parameters). You define parameters just by declaring them as class-level type annotations and setting them in __init__.
* eqx.field(static=True): Some fields are not JAX arrays — integers, strings, booleans. JAX cannot trace through these; they must be compile-time constants. eqx.field(static=True) tells Equinox "this field is not a leaf, treat it as a static constant baked into the computation graph." We need this for n_modes because we use it as a slice index (x_ft[:self.n_modes]), which must be a concrete Python int at trace time.
* Complex arrays in JAX: jnp.fft.rfft returns complex arrays. JAX supports complex arithmetic natively and handles gradients through complex ops correctly using Wirtinger calculus — you don't need to do anything special; jax.grad just works. We store our spectral weights as complex64 arrays directly.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x, mean, std):
    return x * std + mean


class SpectralConv1d(eqx.Module):
    """
    Spectral convolution layer for 1D signals.
    Implements: x-> iFFT(W_k * FFT(x)[:n_modes])
    where W_k is a complex-valued matrix for each Fourier mode k.
   
    Spectral weights stored as two real arrays so Adam sees only real gradients.
    Complex weights are formed inside __call__ and never leave the forward pass.
    """
    w_real: jax.Array  # float32, shape (n_modes, d_in, d_out)
    w_imag: jax.Array  # float32, shape (n_modes, d_in, d_out)
    n_modes: int = eqx.field(static=True)

    def __init__(self, d_in: int, d_out: int, n_modes: int, *, key: jax.Array):
        self.n_modes = n_modes 
        # initialize real and imaginary parts independently with small normal noise.
        # scale by 1/(d_in * d_out) <- à la original FNO paper
        scale = 1.0/(d_in * d_out)
        k1, k2 = jax.random.split(key)
        self.w_real = jax.random.normal(k1, (n_modes, d_in, d_out)) * scale
        self.w_imag = jax.random.normal(k2, (n_modes, d_in, d_out)) * scale

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (nx, d_in) - single sample, no axis

        nx = x.shape[0]

        # form complex weights here, inside forward pass only
        weights = self.w_real + 1j * self.w_imag          # (n_modes, d_in, d_out), complex64


        # ---- Step 1: FFT along the spatial axis -----------------
        # rfft exploits real-valued input: output has shape (nx//2 + 1, d_in)
        # Only positive frequences are kept; the negative ones are conjugate
        # symmetric and irfft reconstructs them automatically

        x_ft = jnp.fft.rfft(x,axis=0)       # (nx//2 + 1, d_in), complex64

        # ------- Step 2: Truncate to the first n_modes Fourier modes ---
        x_ft_trunc = x_ft[:self.n_modes,:]      # (n_modes, d_in)

        # ---- Step 3: complex linear map over modes ------------
        # For each mode k: out_ft[k] = weights[k] @ x_ft_trunc[k]
        # weights[k]: (d_in x d_out) C-valued matrix
        # einsum (Einstein summation) does this for all k

        out_ft = jnp.einsum('ki, kio->ko',
                            x_ft_trunc,
                            weights
        )

        # ----- Step 4: pad back to full frequency ---------------
        # irfft expects (nx//2 + 1) freq bins.
        # high-freq bins with zero are filled (the network learns which low modes matter)

        n_ft = nx // 2 + 1
        out_ft_full = jnp.zeros((n_ft, out_ft.shape[-1]), dtype=jnp.complex64)
        out_ft_full = out_ft_full.at[:self.n_modes, :].set(out_ft)

        # ---- Step 5: Inverse FFT -----------------------
        # n = nx IS mandatory: tells irfft the target signal length
        return jnp.fft.irfft(out_ft_full, n=nx, axis=0)   # (nx, d_out)

"""
# Diagnostic Run

```python
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    layer = SpectralConv1d(d_in=2, d_out=32, n_modes=16, key=key)

    x = jax.random.normal(key, (256, 2))   # (nx=256, d_in=2)
    out = layer(x)

    print("input shape :", x.shape)        # expect (256, 2)
    print("output shape:", out.shape)      # expect (256, 32)
    print("weights dtype:", layer.weights.dtype)   # expect complex64
    print("output dtype :", out.dtype)     # expect float32
    print("output finite:", jnp.all(jnp.isfinite(out)))  # expect True
```
"""

"""
The architecture for the FNOBlock:
Each FNO block has two parallel branches:
```
x ──┬── SpectralConv1d ──────────────┬── (+) ── GELU ── out
    └── pointwise Linear (per point) ─┘
```
The spectral branch captures global structure (mixes information across all spatial locations via Fourier modes).
The pointwise linear branch captures local structure (same matrix applied independently at each spatial point).
Together, they cover both scales.
"""

class FNOBlock(eqx.Module):
    """
    One FNO layer: spectral conv + pointwise lienar, summed, then activation.
    Input and output both have shape (nx, d_v) preserving channel width in the process.
    """
    spectral_conv: SpectralConv1d
    linear: eqx.nn.Linear

    def __init__(self, d_v: int, n_modes: int, *, key:jax.Array):
        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(d_v, d_v, n_modes, key = k1)
        self.linear = eqx.nn.Linear(d_v, d_v, use_bias=True, key = k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (nx, d_v)
        spectral_out = self.spectral_conv(x)
        linear_out = jax.vmap(self.linear)(x)
        return jax.nn.gelu(spectral_out + linear_out)

"""
Diagnostics

```python
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    # Block 1 check (keep from before)
    layer = SpectralConv1d(d_in=2, d_out=32, n_modes=16, key=k1)
    x = jax.random.normal(key, (256, 2))
    out = layer(x)
    print("SpectralConv1d output shape:", out.shape)   # (256, 32)

    # Block 2 check
    block = FNOBlock(d_v=32, n_modes=16, key=k2)
    x2 = jax.random.normal(key, (256, 32))             # (nx, d_v)
    out2 = block(x2)
    print("FNOBlock output shape:", out2.shape)        # (256, 32)
    print("FNOBlock output finite:", jnp.all(jnp.isfinite(out2)))
```

"""

class FNO(eqx.Module):
    """
    Full FNO model for 1D operator learning.
    Maps u0: (nx, ) -> uT: (nX, )
    """
    lifting: eqx.nn.Linear
    blocks: list
    proj1: eqx.nn.Linear
    proj2: eqx.nn.Linear

    def __init__(self, d_v: int, n_modes: int, n_blocks: int, *, key:jax.Array):
        keys = jax.random.split(key, n_blocks + 3)

        # Lifting: (2, ) -> (d_v, ) [input has 2 channels: u0 + grid]
        self.lifting = eqx.nn.Linear(2, d_v, use_bias = True, key = keys[0])

        # FNO Blocks: each maps (nx, d_v) -> (nx, d_v)
        self.blocks = [
            FNOBlock(d_v, n_modes, key=keys[1+i])
            for i in range(n_blocks)
        ] 

        # Projection: (d_v, ) -> (128, ) -> (1, )
        self.proj1 = eqx.nn.Linear(d_v, 128, use_bias = True, key = keys[-2])
        self.proj2 = eqx.nn.Linear(128, 1, use_bias = True, key = keys[-1])

    def __call__(self, u0: jax.Array) -> jax.Array:
        nx = u0.shape[0]

        # -- build spatial grid and concatenate ---
        grid = jnp.linspace(0.0, 1.0, nx)
        x = jnp.stack([u0, grid], axis = -1)

        # --- lifting ----
        x = jax.vmap(self.lifting)(x)

        # ---- FNO Blocks -----
        for block in self.blocks:
            x = block(x)

        # ---- Projection -----
        x = jax.vmap(self.proj1)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.proj2)(x)

        return x.squeeze(-1)

"""
Diagnostics:

```python
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    model = FNO(d_v=32, n_modes=16, n_blocks=4, key=key)

    u0  = jax.random.normal(key, (256,))
    out = model(u0)

    print("input shape :", u0.shape)
    print("output shape:", out.shape)
    print("output finite:", jnp.all(jnp.isfinite(out)))

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"total parameters: {n_params:,}")
```

```python
if __name__ == "__main__":
    from configs.default import Config
    cfg = Config()
    key = jax.random.PRNGKey(0)

    model = FNO(
        d_v=cfg.n_channels,
        n_modes=cfg.n_modes,
        n_blocks=cfg.n_fno_blocks,
        key=key,
    )

    u0  = jax.random.normal(key, (cfg.nx,))   # single sample
    out = model(u0)

    print("input shape :", u0.shape)           # (256,)
    print("output shape:", out.shape)          # (256,)
    print("output finite:", jnp.all(jnp.isfinite(out)))

    # count parameters
    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"total parameters: {n_params:,}")   # expect ~400k–600k
```

```
input shape : (256,)
output shape: (256,)
output finite: True
total parameters: 74,209
```

```
Lifting:          2×32 + 32  =      96
SpectralConv ×4:   4 × 2×(16×32×32) = 131,072   ← w_real + w_imag, both float32
Linear ×4:         4 × (32×32+32)   =   4,224
proj1:             32×128+128       =   4,224
proj2:             128×1+1          =     129
─────────────────────────────────────────────
Total:                               ~139,745
```

"""