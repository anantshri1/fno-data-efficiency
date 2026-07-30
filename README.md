# PDE Solvers based on Spectral Neural Operators 

A data efficiency study benchmarking the performance of Fourier Neural Operators (FNOs) against that of U-Nets for learning the solution operator for the **1-dimensional Burgers' Equation**: $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$, for a given coefficient of viscosity $\nu$.

> Burgers' equation is a fundamental diffusion-convection equation and is the prototype for studying systems that display discontinuities (shock waves). The viscous Burgers' equation can be converted to a linear equation via a Cole-Hopf transformation: $u(x,t) = -2\nu \frac{\partial }{\partial x}ln(\phi(x,t))$, which turns it into the equation: $2\nu \frac{\partial}{\partial x}\bigg[\frac{1}{\phi}\bigg(\frac{\partial \phi}{\partial t} - \nu \frac{\partial^2 \phi}{\partial x^2} \bigg) \bigg] = 0$. This can be integrated with respect to $x$: $\frac{\partial\phi}{\partial t} - \nu \frac{\partial^2\phi}{\partial x^2} = \phi\frac{df(t)}{dt}$, where $\dot{f}$ is a function of time. Introducing the transformation $\phi \to \phi e^f$, the equation reduces to the heat equation: $\frac{\partial \phi}{\partial t} = \nu \frac{\partial^2\phi}{\partial x^2}$. The diffusion equation can be solved at this stage: $u(x,t) = -2\nu \frac{\partial}{\partial x}ln\bigg[\int_{-\infty}^{+\infty}\, exp\bigg(-\frac{(x-x')^2}{4\nu t} - \frac{1}{2\nu}\int_0^{x'}f(x'')dx'' \bigg)dx' \bigg]$.

The project runs in a Python virtual environment. `JAX` is installed in CPU-only mode for development (`jax[cpu]`). The dependency set is intentionally minimal:
* `JAX` and `Equinox` cover the model and all numerical operations. 
* `Optax` covers optimization.
* `MLflow` covers experiment tracking.
* `Matplotlib` covers visualization. 

No higher-level neural operator libraries are used for the model itself; the entire FNO implementation is built from primitives. This is a deliberate choice for two reasons: it demonstrates implementation literacy, and it gives full control over architectural details that matter for the ablation studies.

---
## Data Generation and Basic Configuration 

### A Primer on Gaussian Random Fields
The initial conditions for Burgers' equation are sampled from a *Gaussian Random Field*. A Gaussian Random Field is a probability distribution over functions. When you sample from it, you get a random function defined on your spatial domain. The key property we need is that the sampled functions are **smooth** (no discontinuities, no spikes) and statistically homogeneous (the statistical properties do not depend on position). Both properties follow from how we define the field in Fourier space.

For a periodic domain of length 1 with `nx` grid points, any function can be written as a sum of Fourier modes with wavenumbers `k = 0, 1, 2, ..., nx/2`. The wavenumber `k` tells you how many full oscillations the corresponding mode completes across the domain. `k=0` is the mean, `k=1` is one full oscillation, `k=16` is sixteen oscillations, and so on.

To sample a smooth random function, we work directly in Fourier space. We draw complex Gaussian noise independently at each wavenumber, then scale the noise at each wavenumber by a weight that depends on `k`. If we want smooth functions, we make the weights decay with `k`; high-wavenumber components (fast oscillations) are suppressed, low-wavenumber components (slow oscillations) dominate. The result in physical space is a smooth, random, spatially correlated function.

The specific power spectrum we use is a Gaussian decay: `weight(k) = exp(-0.5 * (length_scale * k)^2)`. The `length_scale` parameter controls how aggressively high modes are suppressed. With `length_scale=0.2`, modes above roughly `k=10` are suppressed by more than one standard deviation, giving functions whose dominant variation occurs over spatial scales larger than about 0.1 times the domain. This is smooth enough to be well-resolved at `nx=256` but complex enough that predicting the solution at time `T` is nontrivial.

After sampling in Fourier space, we apply an inverse real FFT (`irfft`) to get the physical-space function. This is guaranteed to produce a real-valued output because we never explicitly set the negative-frequency components: `irfft` enforces conjugate symmetry automatically. Each sample is then normalized to zero mean and unit standard deviation so that the scale of the initial conditions does not vary across samples. This normalization is purely for numerical convenience and has no effect on the physics.

The GRF approach is standard in the neural operator literature precisely because it gives independent, identically distributed initial conditions with controllable smoothness. Using fixed analytical initial conditions (sinusoids, step functions) would bias the results toward models that happen to fit those specific shapes. The GRF ensures that what we measure is genuine operator learning, not pattern matching to a specific IC family.

### The Burgers' Solver
Burgers equation is: $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$. The left side is a nonlinear advection term; the right side is linear diffusion. At small viscosity ($\nu=0.01$ in this project), the nonlinear term dominates in smooth regions and causes characteristics to converge, steepening gradients until the diffusion term balances them. The result is shock-like structures: **regions of very sharp gradient that are not true discontinuities (the viscosity prevents that) but are numerically challenging**.

We solve this with a pseudo-spectral method using an integrating factor, sometimes called *ETDRK1 (Exponential Time Differencing Runge-Kutta, first order)*. The key insight is that the linear diffusion term, in Fourier space, becomes a simple multiplication: the Fourier coefficient at wavenumber `k` evolves as $\frac{d\hat{u}(k)}{dt} = -\nu k^2 \hat{u}(k)$ from the diffusion term alone. This ODE has an exact solution: $\hat{u}(k, t+dt) = e^{-\nu k^2 d}\hat{u}(k,t)$. We can therefore handle the diffusion term without any time-discretization error by multiplying by this integrating factor at each step.

The nonlinear term $u\frac{\partial u}{\partial x}$ is handled explicitly in physical space. In Fourier space, spatial differentiation is multiplication by $ik$, so $\frac{\partial u}{\partial x}$ transforms to $ik \hat{u}(k)$. The product $u\frac{\partial u}{\partial x}$ is computed by transforming back to physical space, forming the product there, then transforming back. An equivalent and slightly more stable form is to write $u\frac{\partial u}{\partial x} = \frac{1}{2}\frac{\partial u^2}{\partial x}$, which in Fourier space is $ik FFT(\frac{u^2}{2})$.

One full step of the solver is: 
* transform $u$ to Fourier space,
* compute the nonlinear term as described,
* take an explicit Euler step in Fourier space combining the nonlinear term with the integrating factor,
* transform back to physical space. 

This is first-order accurate in time, which is sufficient here because we use a small enough time step (`nt=1000` steps for `t_end=1.0`, giving `dt=0.001`).

The solver is implemented using `jax.lax.scan` rather than a Python for-loop. This is a JAX-specific choice with significant practical consequences. A Python loop over `nt=1000` steps would be unrolled by JAX's `JIT` compiler into a computation graph with 1000 copies of the step function, which takes a long time to compile and produces a very large graph. `jax.lax.scan` compiles the loop as a single recurrent operation, reducing compile time from minutes to seconds and keeping memory usage constant in the number of steps. The tradeoff is that scan requires the loop body to have a fixed signature (`carry, input -> carry, output`) and cannot depend on dynamic values computed during the loop. Our solver satisfies this naturally.

The full configuration used for the rest of the project is shown below:

```python
    pde_nu: float = 0.01          # Burgers' viscosity coefficient
    nx: int = 256                  # spatial resolution (number of grid points). This is our base spatial resolution. The super-resolution test will evaluate at 512 and 1024 — but the model never sees those during training.
    nx_mid: int = 512               # intermediate resolution
    nx_super: int = 1024            # finest resolution
    nt: int = 1000                 # number of time steps in the solver
    nt_super: int = 4000            # time steps in super-rest
    t_end: float = 1.0             # integrate from t=0 to t_end
    n_train_max: int = 2000        # largest training set we'll ever generate
    n_test: int = 200              # held-out test samples (fixed across all runs)
    grf_length_scale: float = 0.2  # controls smoothness of initial conditions: how "wiggly" the initial conditions are — smaller = more high-frequency content = harder problem.
 
    # --- Training ---
    n_epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 3e-4
    lr_decay_rate: float = 0.5
    lr_decay_every: int  = 100
    seed: int = 0
 
    # --- FNO architecture ---
    n_modes: int = 16              # how many Fourier modes to keep, this is the FNO hyperparameter that controls how many Fourier frequencies the spectral layer uses. With nx=256, we have 128 possible modes; we keep only the lowest 16. This is the core truncation that gives FNO its inductive bias.
    n_channels: int = 32           # width of the FNO layers
    n_fno_blocks: int = 4          # depth
    unet_base_channels: int = 15
```

> A critical fix was applied here: the original data generation script saved only `nx=256` data. This is insufficient for the zero-shot super-resolution test, which requires evaluating the trained model on the same initial conditions at finer grids. If we were to generate fresh GRFs at `nx=1024` later, we would be testing on different functions, which is not the experiment.
>
> The correct protocol is to generate and solve at the finest resolution (`nx=1024`), then subsample down to 512 and 256 by taking every 4th and every 2nd spatial point respectively. This guarantees the same underlying function appears at all three discretisations.
>
> The subsampling by striding (taking every k-th point) is valid here because the GRF length scale ensures the functions have no energy above the `nx=256` Nyquist frequency. There is nothing to alias. For rougher fields this would require an explicit low-pass filter before subsampling.

> The other consideration is CFL stability. The **Courant-Friedrichs-Lewy** condition requires that the numerical wave speed times $\frac{dt}{dx}$ must be below a threshold for explicit time-stepping schemes to remain stable. Since `dx = 1/nx`, and we need `dt*max_speed/dx` to be bounded, increasing `nx` by a factor of 4 requires increasing `nt` by the same factor. We set `nt_super=4000` for `nx_super=1024` to satisfy this. Failing to scale `nt` with `nx` would produce a solver that appears to run but generates physically wrong solutions, which would silently corrupt the super-resolution evaluation.

---
## Implementing the FNO

The central conceptual reframe for this project is the distinction between solving a PDE and learning its solution operator. A classical numerical solver takes one specific initial condition `u0` and integrates the PDE forward in time to produce `uT`. It solves one instance at a time and starts from scratch for every new input. An operator learning approach instead treats the entire solution process as a function $\mathcal{G}: u_0 \to u_T$, and trains a neural network to approximate that function across all possible initial conditions simultaneously. Once trained, evaluating the network on a new `u0` requires only a forward pass with no numerical integration. The training data consisting of (u0, uT) pairs generated by the finite-difference Burgers solver, serves as ground truth labels for this supervised learning problem.

### Mathematical Background
A *Fourier Neural Operator* (FNO) is designed to capture the mapping between a continuous input function $X$ and its corresponding continuous output function $Y$ in Fourier
space. To conduct end-to-end training on FNO, function pair $(X , Y)$ are discretized to instance pair $(x, y)$ during the training process. The objective is to learn a mapping $G$
between $(x, y)$, denoted as $y= G(x)$.

The mapping $G$ involves the sequential steps of lifting the input channel using $P$, conducting the mapping through $L$ Fourier layers $\{H_1, H_2,\ldots, H_L \}$, and then projecting back to the original channel through $Q$: $G = Q \circ H_L \circ \ldots \circ H_2 \circ H_1 \circ P$. $P$ and $Q$ are pixel-wise transformations that can be implemented using models like Multilayer perceptron (MLP).

The key architecture of FNO is centred around its **Fourier layer**. Fourier layer typically consists of a pixel-wise linear transformation with weight $W$ and bias $b$, and an integral kernel operator $K$: $H_{basic} = \sigma(Wx + b +K(x))$, with $\sigma$ as the nonlinear activation function, and the integral kernel $K$ undergoing a sequential process involving three operations: Fast Fourier Transformation (FFT), spectral linear transformation, and inverse FFT. The primary parameters of FNO are located in the spectral linear transformation. Hence, to avoid introducing extensive parameters, FNO truncates high-frequency modes in each Fourier layer. These truncated frequency modes can encompass rich spectrum information, especially for high-resolution inputs.

The FNO is specifically designed to learn solution operators for PDEs whose dynamics are naturally expressed in frequency space. Burgers' equation with periodic boundary conditions is a canonical example: the solution is smooth (especially with viscosity $\nu=0.01$), low-frequency Fourier modes carry most of the energy, and the operator mapping $u_0$ to $u_T$ has a compact representation in the spectral domain.

The spectral convolution layer, the core of the FNO here, implements the following sequence:
* Given an input signal `x` of shape `(nx, d_in)`, it applies the real FFT along the spatial axis, producing `(nx//2 + 1, d_in)` complex coefficients.
* It truncates to the first `n_modes=16` coefficients, discarding all high-frequency content.
* It applies an independent complex linear map at each retained mode: for mode `k`, the operation is `out[k] = W_k @ x_ft[k]`, where `W_k` is a learned complex matrix of shape `(d_in, d_out)`. It then zero-pads back to the full `(nx//2 + 1, d_out)` spectrum and applies the inverse real FFT with `n=nx` specified explicitly to recover the target signal length. The zero-padding of discarded modes is the inductive bias. The network is constrained to only modify the lowest $n_{modes}$ frequencies of its input, enforcing the smoothness prior that Burgers' solutions exhibit.

The choice of `rfft` over `fft` is deliberate: real-valued input signals have conjugate-symmetric spectra, meaning the negative frequency components carry no independent information. `rfft` exploits this and returns only the positive-frequency half. `irfft` reconstructs the full real-valued signal from it, enforcing realness automatically. Using the full complex FFT would require manually enforcing Hermitian symmetry, which is a silent failure mode: the model trains, outputs contain a small imaginary part, and the error plateaus mysteriously.

Each FNO block combines two parallel branches:
* The spectral convolution branch captures global spatial structure by mixing information across all locations via Fourier modes.
* A pointwise linear branch, implemented as the same matrix applied independently at each spatial location via `vmap`, captures local structure with no spatial mixing. 

Their outputs are summed and passed through GELU activation. Four such blocks are stacked. A lifting layer maps the two-channel input (`u0` concatenated with a uniform spatial grid on `[0,1]`) up to the internal channel dimension `d_v=32`. A two-layer projection MLP maps back down to a scalar output at each spatial point. Including the spatial grid as a second input channel follows the original FNO paper and gives the model explicit positional information.

The relative L2 error is the standard metric in operator learning. For a batch of predictions, it computes the L2 norm of the prediction error divided by the L2 norm of the true solution, averaged over samples. The denominator normalisation makes the metric scale-invariant across initial conditions with different magnitudes, which is essential when evaluating across a wide range of initial conditions drawn from a GRF.

### Architecture Decisions

The input to the FNO is a single sample `u0` of shape `(nx,)`. The model is defined per-sample; batching is handled externally by `vmap`. This is the idiomatic `JAX` pattern and makes the model independent of batch size. All linear operations inside FNO blocks preserve the channel dimension `d_v`, so the spatial resolution `nx` flows through unchanged and the model is naturally resolution-aware, a property that becomes essential for super-resolution evaluation.

The spatial grid is constructed inside `call` from `jnp.linspace(0, 1, nx)`, not stored as a parameter. This means the model automatically adapts to whatever `nx` is passed at inference time, including resolutions it was never trained on.

The projection head uses a two-layer MLP (`d_v to 128 to 1`) rather than a single linear map. This gives the output a nonlinear readout, which matters because the channel mixing before the scalar output does not need to be constrained to be linear.

### Critical Bug: Complex Weights and Adam

The spectral weights were stored as `complex64` arrays in the `SpectralConv1d` class. This is natural from a mathematical standpoint since the weights operate on complex Fourier coefficients. However, it is incompatible with how `optax`'s `Adam` optimizer handles parameter updates.

Adam maintains two running statistics for each parameter: a first moment (running mean of gradients) and a second moment (running mean of squared gradients). The update rule is: `parameter -= lr * m_t / (sqrt(v_t) + epsilon)`. For real parameters, this is unambiguous. For complex parameters, the squaring operation in the second moment update produces complex values. The square root of a complex number is complex. The resulting updates are numerically broken in a way that is not immediately obvious: the model does learn something (loss decreases from the random baseline), but the optimizer is computing corrupted updates throughout, which manifests as persistent oscillation and a loss floor far above what the architecture is capable of.

The symptom across multiple runs was a test loss that oscillated between 0.5 and 1.0 with no sustained downward trend, regardless of learning rate, decay schedule, or gradient clipping. The correct diagnosis was confirmed by the observation that removing complex weights caused the loss to drop from 0.98 to 0.097 within ten epochs, and reach 0.0224 by epoch 490.

The fix is to store `w_real` and `w_imag` as separate `float32` arrays of shape `(n_modes, d_in, d_out)`, and form the complex weight matrix `weights = w_real + 1j * w_imag` inside call only. The optimizer sees only real arrays with real gradients, moment estimates are well-defined, and convergence is clean. The complex arithmetic still happens correctly in the forward and backward passes through JAX's automatic differentiation, which handles complex operations via Wirtinger calculus.

> The incorrect diagnosis path included: switching from continuous to staircase learning rate decay, reducing learning rate from 1e-3 to 3e-4, adding gradient clipping via `optax.chain`, and adjusting spectral weight initialisation scale. None of these helped because none addressed the underlying optimizer corruption. One change (gradient clipping combined with larger init scale) actively caused divergence, with test loss exceeding 1.0 at some epochs, by introducing gradient norm instability on top of the already-broken moment estimates. The lesson is that oscillating loss with no convergence in a well-structured model should prompt a search for optimizer-parameter type mismatches before hyperparameter tuning.

### Training Infrastructure
The training loop follows the functional `JAX` pattern throughout. There is no in-place mutation of model weights. Each gradient step produces a new model object via `eqx.apply_updates`, which returns a fresh `pytree` with updated array leaves. The original model is not modified. The `PRNG` key is threaded explicitly through training: each epoch derives a fresh shuffle key from the main key via `jax.random.split`, and that key is used by make_batches to shuffle the training set. This makes training fully deterministic from a given seed.

The `train_step` function is decorated with `eqx.filter_jit` rather than raw `jax.jit`. Equinox's `filter_jit` correctly handles the static/dynamic split in module pytrees: non-array fields like `n_modes` (declared with `eqx.field(static=True)`) are treated as compile-time constants, while array fields are traced dynamically. Using raw `jax.jit` with an `eqx.Module` would require handling this split manually.

Gradients are computed via `eqx.filter_value_and_grad`, which wraps `jax.value_and_grad` and knows to differentiate only through array leaves of the model pytree, leaving static fields untouched. The loss function is a closure over the batch data, accepting only the model as its argument, which is the pattern required for `filter_value_and_grad`.

Data normalisation is applied using training set statistics only. Both `u0` and `uT` are independently normalised to zero mean and unit variance using their respective training means and standard deviations. Test data is normalised using training statistics, not test statistics, to avoid data leakage. Normalisation had a meaningful effect on convergence speed and was confirmed working by printing mean and std of normalised arrays before training.

The learning rate schedule uses `optax.exponential_decay` with `staircase=True`. This halves the learning rate every 100 epochs (computed as `lr_decay_every` multiplied by the number of batches per epoch, to convert from epoch units to step units). The staircase flag means the rate steps discretely rather than decaying continuously at every gradient step. Without `staircase=True`, the learning rate drifts continuously, which provides accidental stability at the cost of the schedule being hard to reason about.

`MLflow` logging records hyperparameters `(n_train, n_modes, n_channels, n_fno_blocks, lr, batch_size, n_epochs, seed)` at run start, and `train_rel_l2` and `test_rel_l2` as metrics at each epoch. Checkpoints are saved every 50 epochs and at the end of training using `eqx.tree_serialise_leaves`, which serialises all array leaves of the model pytree to disk. This is the Equinox-native checkpoint format; loading requires a fresh model instance with the same architecture, then `eqx.tree_deserialise_leaves` to populate its weights.

The parameters of the FNO are as follows:
```
Lifting:          2×32 + 32  =      96
SpectralConv ×4:   4 × 2×(16×32×32) = 131,072   ← w_real + w_imag, both float32
Linear ×4:         4 × (32×32+32)   =   4,224
proj1:             32×128+128       =   4,224
proj2:             128×1+1          =     129
─────────────────────────────────────────────
Total:                               ~139,745
```

> All scripts are run as
```bash
python3 -m scripts.script_name
```
> from the repository root. Running scripts directly with `python3 scripts/file.py` adds the scripts directory to `sys.path` rather than the repo root, making cross-package imports `(configs, src)` fail with `ModuleNotFoundError`. The `-m` flag treats the repo root as the top-level package and resolves imports correctly.
>
> The model is defined per-sample. Batching is always handled by `jax.vmap(model)(batch)` at the call site. This keeps the model implementation clean and makes it composable with other JAX transformations.

---
## Implementing the U-Net
<img width="800" height="644" alt="image" src="https://github.com/user-attachments/assets/226b2e7d-6c32-4b2a-a326-fcb8f10c9de8" />

U-net



<img width="800" height="444" alt="image" src="https://github.com/user-attachments/assets/1fa04e29-0d03-4f75-8cea-6d4e0b64af4e" />

---
## **References**
* https://arxiv.org/pdf/2205.10573
* https://arxiv.org/pdf/2404.07200v1
* https://people.esam.northwestern.edu/~chopp/course_notes/446-2.pdf
* https://arxiv.org/pdf/2303.10528
* https://neuraloperator.github.io/dev/theory_guide/fno.html
* https://arxiv.org/pdf/2108.08481
* https://arxiv.org/pdf/2505.11766v1
* https://www.youtube.com/watch?v=COEItKEZ-is
* https://arxiv.org/pdf/2111.00254
* https://arxiv.org/pdf/2010.08895
* https://arxiv.org/pdf/2502.06895
* https://arxiv.org/pdf/1505.04597
* https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
* https://arxiv.org/pdf/2512.01421 (GOAT)
* https://www.physicsx.ai/newsroom/how-a-fourier-neural-operator-learns-to-solve-pdes----and-where-it-falls-short
