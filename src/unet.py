"""
import jax
import jax.numpy as jnp
import equinox as eqx

class UNet(eqx.Module):
    # ======== encoder ================
    enc0: eqx.nn.Conv1d
    down0: eqx.nn.Conv1d
    enc1: eqx.nn.Conv1d
    down1: eqx.nn.Conv1d
    enc2: eqx.nn.Conv1d

    # ========= bottleneck ========
    bottleneck: eqx.nn.Conv1d

    # ========= decoder =============
    up1_proj: eqx.nn.Conv1d
    dec1:     eqx.nn.Conv1d
    up0_proj: eqx.nn.Conv1d
    dec0:     eqx.nn.Conv1d

    # ========== output ============
    out_proj: eqx.nn.Conv1d

    def __init__(self, base_channels: int=32, *, key: jax.Array):
        C = base_channels
        keys = jax.random.split(key, 11)

        # NEW THING 1 — eqx.nn.Conv1d operates on (channels, length), not (length, channels).
        # This is the standard JAX/PyTorch convention for convolutions.
        # The FNO used (length, channels) because we were doing manual einsums.
        # Here we just follow Conv1d's native layout.

        # padding = 1, k = 3 preserves spatial length: L_out = L_in.
        self.enc0 = eqx.nn.Conv1d(1, C, 3, padding = 1, key = keys[0])

        # stided conv for downsampling: k = 2, stride = 2 -> L_out = L_in //2.
        # No padding needed; k = 2 tiles exactly into even-length inputs.

        self.down0 = eqx.nn.Conv1d(C, C, 2, stride = 2, key = keys[1])

        self.enc1 = eqx.nn.Conv1d(C, 2*C, 3, padding = 1, key=keys[2])

        self.down1 = eqx.nn.Conv1d(2*C, 2*C, 2, stride = 2, key = keys[3])

        self.enc2 = eqx.nn.Conv1d(2*C, 4*C, 3, padding = 1, key = keys[4])

        self.bottleneck = eqx.nn.Conv1d(4*C, 4*C, 3, padding = 1, key = keys[5])

        # NEW THING 2 — ConvTranspose1d is the learnable inverse of a strided Conv1d.
        # Conv1d(k=2, stride=2) halves length; ConvTranspose1d(k=2, stride=2) doubles it.
        # L_out = (L_in - 1) * stride + kernel_size = 2 * L_in (for k=2, s=2).
        # Weight shape here is (in_channels, out_channels, kernel_size) —
        # flipped from Conv1d's (out_channels, in_channels, kernel_size).

        self.up1_proj = eqx.nn.Conv1d(4*C, 2*C, 1, key=keys[6])

        # After up1 + skip concat: 2C (from up1) + 2C (from enc1) = 4C input channels.
        self.dec1 = eqx.nn.Conv1d(4*C, 2*C, 3, padding=1, key=keys[7])

        self.up0_proj = eqx.nn.Conv1d(2*C, C, 1, key=keys[8])

        # After up0 + skip concat: C + C = 2C input channels.
        self.dec0 = eqx.nn.Conv1d(2*C, C, 3, padding=1, key=keys[9])

        # k=1 pointwise conv: mixes C channels → 1, no spatial mixing.
        self.out_proj = eqx.nn.Conv1d(C, 1, 1, key=keys[10])

    def __call__(self, u0: jax.Array) -> jax.Array:
        # u0: (nx, ) - single sample, same as FNO

        # NEW THING 3 — adding the channel dimension.
        # Conv1d wants (channels, length). Our input has no channel axis.
        # u0[None, :] inserts a length-1 axis at position 0: (nx,) → (1, nx).
        x = u0[None, :]                              # (1, nx)

        # --- Encoder: compress spatial resolution, store skip features ---
        skip0 = jax.nn.gelu(self.enc0(x))           # (C, nx)
        x     = self.down0(skip0)                   # (C, nx//2)
        skip1 = jax.nn.gelu(self.enc1(x))           # (2C, nx//2)
        x     = self.down1(skip1)                   # (2C, nx//4)
        x     = jax.nn.gelu(self.enc2(x))           # (4C, nx//4)

        # --- Bottleneck ---
        x = jax.nn.gelu(self.bottleneck(x))         # (4C, nx//4)

        # --- Decoder: expand back, injecting skip features at each scale ---
        x = jnp.repeat(x, 2, axis=-1)              # (4C, nx//2)
        x = self.up1_proj(x)                        # (2C, nx//2)
        # jnp.concatenate along axis=0 because axis 0 is the channel axis.
        # This is the skip connection: paste the encoder's nx//2 features
        # alongside the decoder's upsampled features.
        x = jnp.concatenate([x, skip1], axis=0)    # (4C, nx//2)
        x = jax.nn.gelu(self.dec1(x))              # (2C, nx//2)

        x = jnp.repeat(x, 2, axis=-1)              # (2C, nx)
        x = self.up0_proj(x)                        # (C, nx)
        x = jnp.concatenate([x, skip0], axis=0)    # (2C, nx)
        x = jax.nn.gelu(self.dec0(x))              # (C, nx)

        # --- Output ---
        x = self.out_proj(x)                        # (1, nx)
        return x.squeeze(0)                         # (nx,) — matches FNO's output shape


Diagnostics
```python
import sys; sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
from src.utils import count_params

key = jax.random.PRNGKey(0)
model = UNet(base_channels=32, key=key)

u0  = jax.random.normal(key, (256,))
out = model(u0)

print("input shape :", u0.shape)           # (256,)
print("output shape:", out.shape)          # (256,) — must match FNO
print("output finite:", jnp.all(jnp.isfinite(out)).item())  # True
print("param count :", count_params(model))  # expect ~142,081
```
"""

# src/unet.py

import jax
import jax.numpy as jnp
import equinox as eqx


class UNet(eqx.Module):
    """
    1D U-Net baseline, 5 downsampling levels.

    Resolution ladder (nx=256):
        256 -> 128 -> 64 -> 32 -> 16 -> 8

    Channel schedule [C, 2C, 4C, 4C, 4C]: growth is capped at 4C because
    standard doubling (up to 16C) would cost ~400k params — far over budget.
    Depth does the receptive-field work; width is held back to pay for it.

    Receptive field: ~220 of 256 input points (~86% of the domain).
    The previous 2-level version had RF=32 (12.5%) and structurally could not
    see enough of u0 to predict uT, since GRF length_scale=0.2 alone spans
    ~51 points and Burgers advection widens the true dependence further.

    Requires nx divisible by 32. Holds for 256, 512, 1024.
    """
    # Encoder
    enc0: eqx.nn.Conv1d
    n_e0: eqx.nn.GroupNorm
    down0: eqx.nn.Conv1d
    enc1: eqx.nn.Conv1d
    n_e1: eqx.nn.GroupNorm
    down1: eqx.nn.Conv1d
    enc2: eqx.nn.Conv1d
    n_e2: eqx.nn.GroupNorm
    down2: eqx.nn.Conv1d
    enc3: eqx.nn.Conv1d
    n_e3: eqx.nn.GroupNorm
    down3: eqx.nn.Conv1d
    enc4: eqx.nn.Conv1d
    n_e4: eqx.nn.GroupNorm
    down4: eqx.nn.Conv1d
    # Bottleneck
    bottleneck: eqx.nn.Conv1d
    n_bn: eqx.nn.GroupNorm
    # Decoder
    up4_proj: eqx.nn.Conv1d
    dec4: eqx.nn.Conv1d
    n_d4: eqx.nn.GroupNorm
    up3_proj: eqx.nn.Conv1d
    dec3: eqx.nn.Conv1d
    n_d3: eqx.nn.GroupNorm
    up2_proj: eqx.nn.Conv1d
    dec2: eqx.nn.Conv1d
    n_d2: eqx.nn.GroupNorm
    up1_proj: eqx.nn.Conv1d
    dec1: eqx.nn.Conv1d
    n_d1: eqx.nn.GroupNorm
    up0_proj: eqx.nn.Conv1d
    dec0: eqx.nn.Conv1d
    n_d0: eqx.nn.GroupNorm
    # Output
    out_proj: eqx.nn.Conv1d

    def __init__(self, base_channels: int = 15, *, key: jax.Array):
        C = base_channels
        k = jax.random.split(key, 22)

        # GroupNorm with groups=1 normalises over (all channels, spatial) jointly
        # — i.e. LayerNorm. Chosen over groups=8 so we never hit a
        # "channels must be divisible by groups" error when C is tuned for the
        # parameter budget. Costs 2 params per channel; negligible.
        def norm(ch):
            return eqx.nn.GroupNorm(groups=1, channels=ch)

        # --- Encoder ---
        self.enc0  = eqx.nn.Conv1d(1,   C,   3, padding=1, key=k[0]);  self.n_e0 = norm(C)
        self.down0 = eqx.nn.Conv1d(C,   C,   2, stride=2,  key=k[1])

        self.enc1  = eqx.nn.Conv1d(C,   2*C, 3, padding=1, key=k[2]);  self.n_e1 = norm(2*C)
        self.down1 = eqx.nn.Conv1d(2*C, 2*C, 2, stride=2,  key=k[3])

        self.enc2  = eqx.nn.Conv1d(2*C, 4*C, 3, padding=1, key=k[4]);  self.n_e2 = norm(4*C)
        self.down2 = eqx.nn.Conv1d(4*C, 4*C, 2, stride=2,  key=k[5])

        self.enc3  = eqx.nn.Conv1d(4*C, 4*C, 3, padding=1, key=k[6]);  self.n_e3 = norm(4*C)
        self.down3 = eqx.nn.Conv1d(4*C, 4*C, 2, stride=2,  key=k[7])

        self.enc4  = eqx.nn.Conv1d(4*C, 4*C, 3, padding=1, key=k[8]);  self.n_e4 = norm(4*C)
        self.down4 = eqx.nn.Conv1d(4*C, 4*C, 2, stride=2,  key=k[9])

        # --- Bottleneck (nx/32) ---
        self.bottleneck = eqx.nn.Conv1d(4*C, 4*C, 3, padding=1, key=k[10])
        self.n_bn = norm(4*C)

        # --- Decoder ---
        # Each level: jnp.repeat upsamples, 1x1 conv adjusts channels,
        # concat injects the skip, 3-tap conv refines.
        # dec input channels = (proj output) + (skip channels).
        self.up4_proj = eqx.nn.Conv1d(4*C, 4*C, 1, key=k[11])
        self.dec4 = eqx.nn.Conv1d(8*C, 4*C, 3, padding=1, key=k[12]); self.n_d4 = norm(4*C)

        self.up3_proj = eqx.nn.Conv1d(4*C, 4*C, 1, key=k[13])
        self.dec3 = eqx.nn.Conv1d(8*C, 4*C, 3, padding=1, key=k[14]); self.n_d3 = norm(4*C)

        self.up2_proj = eqx.nn.Conv1d(4*C, 4*C, 1, key=k[15])
        self.dec2 = eqx.nn.Conv1d(8*C, 4*C, 3, padding=1, key=k[16]); self.n_d2 = norm(4*C)

        self.up1_proj = eqx.nn.Conv1d(4*C, 2*C, 1, key=k[17])
        self.dec1 = eqx.nn.Conv1d(4*C, 2*C, 3, padding=1, key=k[18]); self.n_d1 = norm(2*C)

        self.up0_proj = eqx.nn.Conv1d(2*C, C,   1, key=k[19])
        self.dec0 = eqx.nn.Conv1d(2*C, C,   3, padding=1, key=k[20]); self.n_d0 = norm(C)

        self.out_proj = eqx.nn.Conv1d(C, 1, 1, key=k[21])

    def __call__(self, u0: jax.Array) -> jax.Array:
        # u0: (nx,) -> (nx,), same external interface as FNO
        x = u0[None, :]                                   # (1, nx)

        # --- Encoder: store pre-downsample features as skips ---
        s0 = jax.nn.gelu(self.n_e0(self.enc0(x)))         # (C,   nx)
        x  = self.down0(s0)                               # (C,   nx/2)
        s1 = jax.nn.gelu(self.n_e1(self.enc1(x)))         # (2C,  nx/2)
        x  = self.down1(s1)                               # (2C,  nx/4)
        s2 = jax.nn.gelu(self.n_e2(self.enc2(x)))         # (4C,  nx/4)
        x  = self.down2(s2)                               # (4C,  nx/8)
        s3 = jax.nn.gelu(self.n_e3(self.enc3(x)))         # (4C,  nx/8)
        x  = self.down3(s3)                               # (4C,  nx/16)
        s4 = jax.nn.gelu(self.n_e4(self.enc4(x)))         # (4C,  nx/16)
        x  = self.down4(s4)                               # (4C,  nx/32)

        # --- Bottleneck ---
        x = jax.nn.gelu(self.n_bn(self.bottleneck(x)))    # (4C,  nx/32)

        # --- Decoder ---
        x = self.up4_proj(jnp.repeat(x, 2, axis=-1))      # (4C,  nx/16)
        x = jax.nn.gelu(self.n_d4(self.dec4(jnp.concatenate([x, s4], axis=0))))

        x = self.up3_proj(jnp.repeat(x, 2, axis=-1))      # (4C,  nx/8)
        x = jax.nn.gelu(self.n_d3(self.dec3(jnp.concatenate([x, s3], axis=0))))

        x = self.up2_proj(jnp.repeat(x, 2, axis=-1))      # (4C,  nx/4)
        x = jax.nn.gelu(self.n_d2(self.dec2(jnp.concatenate([x, s2], axis=0))))

        x = self.up1_proj(jnp.repeat(x, 2, axis=-1))      # (2C,  nx/2)
        x = jax.nn.gelu(self.n_d1(self.dec1(jnp.concatenate([x, s1], axis=0))))

        x = self.up0_proj(jnp.repeat(x, 2, axis=-1))      # (C,   nx)
        x = jax.nn.gelu(self.n_d0(self.dec0(jnp.concatenate([x, s0], axis=0))))

        return self.out_proj(x).squeeze(0)                # (nx,)

import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp
from src.unet import UNet
from src.utils import count_params

model = UNet(base_channels=15, key=jax.random.PRNGKey(0))
out = model(jax.random.normal(jax.random.PRNGKey(1), (256,)))
print("shape:", out.shape, "| finite:", jnp.all(jnp.isfinite(out)).item())
print("params:", count_params(model))     # expect ~150k