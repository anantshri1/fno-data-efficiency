"""
What we need to build in src/data_gen.py:
* A GRF (Gaussian Random Field) sampler — generates random but smooth initial conditions u(x, 0)
* A Burgers' pseudo-spectral solver — takes an initial condition and integrates forward to u(x, T)
* A dataset builder — wraps both, produces (input, output) pairs and saves them

On the solver choice: We'll use a pseudo-spectral method with an integrating factor. This is worth understanding:
* The diffusion term ν ∂²u/∂x² is linear → in Fourier space it becomes -ν k² û, which we can solve exactly (no time-step error on this term)
* The nonlinear term u ∂u/∂x we handle explicitly in physical space
* This is the natural solver to write when you're building an FNO, because you're already thinking in Fourier space
"""

import jax
import jax.numpy as jnp

def sample_grf(key, nx, length_scale, n_samples = 1):
    """
    Sample from a Gaussian Random Field on [0,1) with periodic BCs.
    Strategy:
        - Build a power spectrum that decays with wavenumber (smooth fields)
        - Sample Gaussian noise in Fourier space, scaled by the spectrum
        - iFFT back to physical space (automatically periodic)
    
    Args:
        key             : JAX PRNG Key
        nx              : number of spatial grid points
        length_scale    : controls smoothness; larger = smoother
        n_samples       : how many independent fields to draw

    Returns:
        u0: array of shape (n_samples, nx), real-valued
    """

    # wavenumbers for a grid of size nx
    k = jnp.fft.rfftfreq(nx, d = 1.0/nx)   # Returns the Discrete Fourier Transform sample frequencies.
    # shape: (nx // 2 + 1)

    # power spectrum: Gaussian decay in frequency space
    # fields with high wavenumber (fast oscillations) are suppressed
    power = jnp.exp(-0.5*(length_scale * k)**2)

    # sample complex Gaussian noise in Fourier space
    key_real, key_imag = jax.random.split(key)
    noise_real = jax.random.normal(key_real, shape=(n_samples, nx//2 + 1))
    noise_imag = jax.random.normal(key_imag, shape=(n_samples, nx//2 + 1))
    u0_hat = (noise_real + 1j * noise_imag) * power     # shape: (n_samples, nx//2 + 1)

    # back to physical space
    u0 = jnp.fft.irfft(u0_hat, n=nx)    # shape: (n_samples, nx), real

    # normalize to standard Normal
    mean = u0.mean(axis=-1, keepdims=True)
    std = u0.std(axis=-1, keepdims = True)
    u0 = (u0-mean)/(std+1e-8)

    return u0

def solve_burgers(u0, nu, t_end, nt):
    """
    Solve Burgers' equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    using a pseudo-spectral method with integrating factor (ETDRK1).

    The key idea in Fourier space:
      û' = -ν k² û  (linear/diffusion, solved exactly)
            - FFT(u * IFFT(ik û))  (nonlinear, handled explicitly)

    We absorb the linear term into an integrating factor:
      v = exp(ν k² t) û   →   removes stiffness from diffusion

    Args:
        u0    : initial condition, shape (nx,)
        nu    : viscosity
        t_end : end time
        nt    : number of time steps

    Returns:
        u_final : solution at t=t_end, shape (nx,)
    """

    nx = u0.shape[0]
    dt = t_end/nt

    # wavenumbers: k = [0,1,2, ..., nx/2] for rfft output
    k = jnp.fft.rfftfreq(nx, d = 1.0/nx)
    ik = 1j * k     # derivative in Fourier space
    lin = -nu * k**2    # linear term coefficient

    # integrating factors and its inverse
    E = jnp.exp(lin * dt)
    u0_hat = jnp.fft.rfft(u0)       # initial condition in Fourier space

    def step(u_hat, _):
        """ Single RK1 step in Fourier space with integrating factor. """
        u = jnp.fft.irfft(u_hat, n=nx)      # to physical space
        nonlinear = -0.5 * jnp.fft.rfft(u**2)       # FFT of -0.5 * d(u²)/dx ...
        nonlinear = ik * nonlinear      # ... times ik gives -u * du/dx
        u_hat_new = E * (u_hat + dt * nonlinear) 
        return u_hat_new, None

    # jax.lax.scan replaces a Python for-loop with a compiled scan
    # signature: scan(f, init_carry, xs) -> (final_carry, stacked_outputs)
    # xs = None with length=nt means "run nt steps, no per-step input"

    u_final_hat, _ = jax.lax.scan(step, u0_hat, None, length = nt)

    return jnp.fft.irfft(u_final_hat, n = nx)

def generate_dataset(key, n_samples, cfg):
    """
    Generate (initial_condition, solution) pairs for Burgers' equation.

    Args:
        key             : JAX PRNG key
        n_samples       : how many pairs to generate
        cfg             : config instance

    Returns:
        u0          : initial conditions, shape (n_samples, nx)
        uT          : solutions at t=t_end, shape (n_samples. nx)
    """
    # sample all initial conditions at once
    u0 = sample_grf(key, nx = cfg.nx, length_scale=cfg.grf_length_scale,
                    n_samples=n_samples)

    # vmap solve_burgers over batch dimension
    # vmap vectorizes solve_burgers
    solve_batch = jax.vmap(
        lambda u: solve_burgers(u, nu = cfg.pde_nu, t_end = cfg.t_end, nt=cfg.nt)
    )
    uT = solve_batch(u0)

    return u0, uT

import os
import numpy as np

def save_dataset(u0, uT, path):
    """Save dataset as a compressed numpy archive."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, u0=np.array(u0), uT=np.array(uT))
    print(f"Saved {u0.shape[0]} samples to {path}")

def load_dataset(path):
    """Load dataset from a compressed numpy archive."""
    data = np.load(path)
    return jnp.array(data['u0']), jnp.array(data['uT'])

def generate_dataset_superres(key, n_samples, cfg):
    """
    Generate test data for zero-shot super-resolution evaluation.

    Protocol: sample GRF and solve at nx_super (finest resolution),
    then subsample down to nx_mid and nx_base. This ensures the same
    underlying function appears at all three discretisations.

    Args:
        key      : JAX PRNG key
        n_samples: number of samples (typically n_test)
        cfg      : Config instance

    Returns:
        (u0_base, uT_base) : at nx=256
        (u0_mid,  uT_mid)  : at nx=512
        (u0_sup,  uT_sup)  : at nx=1024
    """
    # Sample and solve at finest resolution
    u0_sup = sample_grf(key, nx=cfg.nx_super,
                        length_scale=cfg.grf_length_scale,
                        n_samples=n_samples)

    solve_batch = jax.vmap(
        lambda u: solve_burgers(u, nu=cfg.pde_nu, t_end=cfg.t_end, nt=cfg.nt_super)
    )
    uT_sup = solve_batch(u0_sup)

    # Subsample by taking every k-th point along the spatial axis
    f_mid  = cfg.nx_super // cfg.nx_mid   # = 2
    f_base = cfg.nx_super // cfg.nx       # = 4

    u0_mid  = u0_sup[:, ::f_mid]
    uT_mid  = uT_sup[:, ::f_mid]
    u0_base = u0_sup[:, ::f_base]
    uT_base = uT_sup[:, ::f_base]

    return (u0_base, uT_base), (u0_mid, uT_mid), (u0_sup, uT_sup)


def save_dataset_superres(base, mid, sup, path):
    """
    Save all three resolutions into a single compressed archive.
    Keys: u0_256, uT_256, u0_512, uT_512, u0_1024, uT_1024
    """
    (u0_base, uT_base) = base
    (u0_mid,  uT_mid)  = mid
    (u0_sup,  uT_sup)  = sup

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        u0_256=np.array(u0_base),  uT_256=np.array(uT_base),
        u0_512=np.array(u0_mid),   uT_512=np.array(uT_mid),
        u0_1024=np.array(u0_sup),  uT_1024=np.array(uT_sup),
    )
    print(f"Saved superres dataset ({u0_base.shape[0]} samples, "
          f"3 resolutions) to {path}")