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
## Implementing the FNO in `JAX`

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
## A Training Agnostic Baseline: the U-Net

The choice to build a U-Net rather than a plain CNN is motivated by the following reasoning: a plain CNN has a local receptive field determined by its kernel size and depth. A seven-layer CNN with `k=3` has a receptive field of 15 pixels, which on a 256-point grid sees 6% of the domain. Beating a model that sees 6% of the domain with an FNO that sees 100% of it does not constitute a controlled experiment. The U-Net earns its place as the primary baseline because downsampling, in principle, gives it a global receptive field: each spatial position in the bottleneck aggregates information from a large window of the input. The remaining structural difference between an FNO and a U-Net with sufficient depth is then specifically spectral decomposition versus learned multi-scale spatial filters. 

Four architectural decisions were key here:
* Downsampling was done with strided `Conv1d` rather than average pooling. A strided convolution with kernel size 2 and stride 2 halves the spatial dimension and is learnable, meaning the network can choose how to compress. This adds parameters to the downsampling layers but keeps the compression itself part of the learned representation.
* Upsampling was intended to use `ConvTranspose1d`, the standard learnable inverse of a strided convolution. A `ConvTranspose1d` with kernel size 2 and stride 2 should double the spatial dimension according to the formula `L_out` equals `(L_in minus 1) times stride plus kernel_size`, giving `2 times L_in`.
* Skip connections concatenate encoder feature maps into the decoder at the matching resolution. When the decoder has upsampled back to `nx/2`, it concatenates the encoder's feature map from the `nx/2` level, which was stored before the stride-2 downsample. The channel count after concatenation therefore doubles: if the decoder upsampled to 2C channels and the encoder skip has 2C channels, the concatenated tensor fed to the decoder conv has 4C channels. Getting these channel counts right in both the field declarations and the constructor is where most of the bookkeeping lives.
* `GroupNorm` was added at every convolutional layer in the five-level version. The choice of `groups=1` means the normalisation is computed over all channels and all spatial positions jointly for each sample, which is equivalent to `LayerNorm`. The alternative, `groups=8`, was rejected because it requires the channel count to be divisible by 8, and the channel count is tuned to hit a parameter budget rather than chosen for divisibility. `groups=1` costs two parameters per channel (a learned scale and bias) and is negligible in the total count.

The parameter budget target was approximately 139,745 to match the FNO. With the channel schedule `[C, 2C, 4C, 4C, 4C]` and five downsampling levels, the total parameter count scales roughly as 656 times C squared, and `C=15` gives approximately 150k, which is 7% above the FNO. The decision was to accept this and err on the side of giving the baseline more capacity rather than less. Both param counts (139,745 and 149,641) are reported explicitly and without rounding in all results. The FNO wins with fewer parameters, which strengthens rather than weakens the conclusion.

### Infrastructure
The training script decision was to write a new `scripts/train_model.py` rather than modify the working script. `train_fno.py` remains untouched as a reference; `train_model.py` is structurally identical to it except that the model construction is routed through a `build_model` function that takes a `model_type` string and returns either an FNO or a UNet. Everything else, the data loading, normalisation using training statistics only, the staircase LR schedule, the MLflow logging, the per-epoch checkpoint cadence, is shared code. This matters for experimental integrity: any difference in results between the two models is attributable to the model, not to differing training conditions.

The CLI exposes 
```bash
--model (required, fno or unet), --n_train (optional, defaults to full 2000), and --seed (optional, overrides cfg.seed)
```
The convention `model_type_Nn_train_seedseed_final.eqx` for checkpoint names was established here for consistency, since the sweep script will check for checkpoint existence to decide whether to skip a run.

The results of the first run are shown below (`seed 0, N=2000, 500 epochs`):

|Model	|Params|	Train rel-L2|	Test rel-L2|
|------|------|----------------|---------------|
|FNO	|139,745|	0.0109|	0.0224|
|U-Net	|149,641|	0.0226	|0.0716|

### Critical Bug with `ConvTranspose1d`

The first version of the U-Net used `eqx.nn.ConvTranspose1d` for upsampling. After writing the full file and running diagnostics, the forward pass crashed with a concatenation error: arrays with shapes `(64, 64)` and `(64, 128)` cannot be concatenated along `axis 0`. The first array is the upsampled decoder output; the second is the encoder skip. The spatial dimensions do not match, which means the `ConvTranspose1d` did not actually double the length.

The hypothesis was that `JAX`'s underlying `conv_transpose` operation has ambiguous padding semantics, and that `Equinox`'s integer padding argument maps to something that produces `L_out` equal to `L_in` rather than `2 x L_in`. This is a documented source of confusion: the JAX padding parameter for `conv_transpose` expects a string or a sequence of tuples describing the padding on both sides, and integer padding does not translate the way `Conv1d`'s integer padding does. Rather than debugging the padding arithmetic, the decision was to replace the `ConvTranspose` entirely with an explicit spatial resize followed by a learnable channel projection. `jnp.repeat(x, 2, axis=-1)` duplicates each spatial position: `[a, b, c]` becomes `[a, a, b, b, c, c]`, exactly doubling the length along the last axis. A `1x1` `Conv1d` then handles the channel reduction. This approach separates the two concerns, spatial resizing and channel mixing, which is arguably cleaner than `ConvTranspose` regardless of the padding issue. The dec convolution that follows then refines the features. This pattern, repeat then project then refine, is what the final architecture uses throughout the decoder.

### Failure of the Receptive Field
With the `ConvTranspose` bug fixed, the two-level U-Net passed all shape checks and produced finite output. The diagnostic check at `N=2000, 500 epochs` gave test `rel-L2` of `0.49` and a train loss of approximately `0.56`. These two numbers together are the diagnostic.

A large train/test gap indicates overfitting or data scarcity: the model memorises training samples but does not generalise. The numbers here showed the opposite: train and test losses were nearly identical, both around 0.5. A 131k-parameter model that cannot overfit 2000 training samples is not learning. It is underfitting, meaning it does not have the capacity to represent the target function, regardless of how much data it is shown.

The cause is the receptive field. Global receptive field is a **property of having enough downsampling layers**. The receptive field can be computed exactly using the recursion `RF_out equals RF_in plus (k minus 1) times jump_in`, where jump doubles at every stride-2 layer. For the two-level U-Net:

```
enc0, k=3, stride=1, jump=1: RF = 3
down0, k=2, stride=2, jump=1: RF = 4, jump becomes 2
enc1, k=3, stride=1, jump=2: RF = 8
down1, k=2, stride=2, jump=2: RF = 10, jump becomes 4
enc2, k=3, stride=1, jump=4: RF = 18
bottleneck, k=3, stride=1, jump=4: RF = 26
dec1, k=3, stride=1: RF = 30
dec0, k=3, stride=1: RF = 32
```

The receptive field of the final output is 32 pixels out of 256. This means each predicted output point depends on at most 32 input points, covering 12.5% of the domain.

This is insufficient for two independent reasons:
* First, the GRF initial conditions have `length_scale=0.2` on a unit periodic domain, meaning input features are spatially correlated over approximately `0.2 times 256 equals 51 pixels`. A model that can only see 32 pixels cannot fully resolve the input structure it is trying to learn from.
* Second, Burgers' equation advects information: over `t_end=1.0` with `O(1)` velocities, characteristics travel significant fractions of the domain, meaning the value of `uT` at a point depends on `u0` over a window much wider than the immediate neighbourhood. 0.49 is the best achievable error for a model doing local-window regression on a problem with non-local dependencies.

The fix is depth: five downsampling levels bring the resolution to 256/32 equals 8 at the bottleneck, and the receptive field calculation gives approximately 220 pixels, covering 86% of the domain. At this depth the architecture genuinely has global context. The channel schedule is adjusted to `[C, 2C, 4C, 4C, 4C]` because standard doubling to 16C at five levels would cost roughly 400k parameters, three times the FNO budget, which would make the comparison meaningless. Capping at 4C keeps the budget controlled while letting depth do the work. The constraint that nx must be divisible by 32 holds for all three resolutions in the dataset: 256, 512, and 1024.

The corrected architecture trained to 0.0716 at N=2000, with train at 0.023. **The U-Net is overfitting more at the same data volume, and the FNO's frequency-domain inductive bias acts as an implicit regulariser at matched budget.**

---
## Benchmarking FNO and U-Net performance

### Infrastructure
The sweep script runs each of the 60 grid configurations as a separate subprocess rather than in a loop within a single Python process. The alternative was to import the training function and call it 60 times in-process. The subprocess approach was chosen for two reasons:
* First, each run gets a clean `JAX` and `XLA` state, including a fresh `JIT` compilation cache with no residue from prior runs.
* Second, a crash in one run cannot affect the other 59, because each process is isolated. The cost is Python interpreter startup overhead per run, which is negligible against 500 training epochs.

The skip logic checks whether the final checkpoint file for a given configuration already exists and skips it if so. This makes the sweep resumable: if a Colab tunnel disconnects or a machine sleeps, rerunning the script picks up where it left off. The script also accepts a `--dry_run` flag that prints the planned commands without executing them, useful for verifying the grid before committing CPU time, and a `--force` flag that ignores existing checkpoints and re-runs everything.

Per-run `stdout` and `stderr` are redirected to individual log files under `logs/sweep/`, one per configuration. Failed runs are logged and skipped rather than aborting the sweep. The outer summary at the end of the script reports success, skipped, and failed counts with paths to any failure logs.

For unattended execution the recommended invocation is: `mkdir -p logs/sweep && nohup python -u -m scripts.sweep > logs/sweep_main.log 2>&1 &`. The `-u` flag forces unbuffered output so that tail `-f` on the log file shows progress in real time rather than dumping in blocks.

The sweep ran on local CPU. Wall time for the full 60-run grid was approximately 4 hours.

### Analysis Pipeline
The analysis script pulls results from MLflow and applies several filters and transformations before producing any output.
* The `load` step queries the MLflow experiment by name and casts the string-typed parameter columns to their correct types. The filter requires `model_type` to be either `"fno"` or `"unet"`, `n_train` to be in the N grid, and seed to be in 0 through 4. This excludes the earlier reference run (which was logged by a script that never records `model_type` and therefore has a null in that column), the LR-decay control run (seed 999, outside the seed range), and any other exploratory runs that happened to share in-grid N and seed values.
* The `deduplication` step keeps the most recent run per `(model_type, n_train, seed)` triplet, sorting by `start_time`. 
* The `summarize` step aggregates over seeds within each cell, computing mean, standard deviation, and count of both test and train relative L2 error. A gap column is added as test mean minus train mean.
* The `audit` step prints all distinct `n_params` values per model after deduplication. This is the sharpest provenance check: the FNO must show exactly 139745 and the U-Net must show exactly 149641. Any other value in either group means a run from a different architecture survived deduplication and is corrupting the analysis.
* The `n_for_target_error` function implements log-log linear interpolation. Given a target error value and a model, it finds the two adjacent grid points that bracket the target, fits a straight line in log-log space, and solves for the N value where the line crosses the target. The assumption is that error follows a local power law, error approximately equal to $A N^{-\alpha}$, with $\alpha$ constant inside the interval. The function explicitly refuses to extrapolate: if the target error falls outside the model's observed range it raises an error rather than guessing. Values should be quoted with limited precision — approximately 300 and approximately 1250, not 307 and 1254, since the latter implies measurement precision that is not present in an interpolated estimate.
* The `bootstrap` estimates a confidence interval on the multiplier. It resamples the five seed values within each (model, N) cell independently with replacement, rebuilds the full mean curve for both models from the resampled values, re-interpolates both crossing points, and computes their ratio. This is repeated 2000 times. The 2.5th and 97th percentiles of the resulting distribution are the 95% confidence interval. Resampled curves occasionally become non-monotonic or fail to bracket the target, in which case that sample is skipped. The function reports how many samples were skipped out of how many attempted. The confidence interval covers seed variance only — it says nothing about data-draw variance, epoch-budget choices, or target-error selection. With only five seeds the bootstrap is coarse, and the interval should be described as five-seed bootstrap rather than implying asymptotic coverage guarantees.

### Debugging Duplicate Runs
When the analysis was first run it pulled 121 runs for 60 grid cells. The summary showed most cells had ten seeds rather than five, and two cells had anomalous counts: FNO at N=100 had exactly five while everything else had ten, and U-Net at N=100 had six. The U-Net N=2000 cell had a standard deviation of 0.139 on a mean of 0.066, where every other cell had standard deviations in the range 0.001 to 0.010.

The diagnosis function revealed that nearly every configuration had two MLflow runs with identical metrics, start times roughly 40 to 90 seconds apart, in a consistent interleaved pattern across the whole grid. Since training is deterministic — same code, same config, same seed always produces bit-identical output — these were genuine independent re-runs of the same config landing in the same MLflow experiment, not a double-logging bug. The most likely cause was two `sweep.py` processes running concurrently: the first `nohup` invocation failed on a missing `logs/` directory, a second was launched while the first may have still been alive, and both walked the grid in the same order with a timing offset.

This pattern is harmless for results but does mean the raw run count is meaningless on its own. The deduplication step resolves it by keeping only the most recent run per configuration.

> The U-Net `N=2000` anomaly was a different problem. One of the duplicate runs for that cell showed `base_channels=32` and test error 0.504 — a value that matches the broken two-level U-Net from an early iteration, which had a receptive field of only 32 out of 256 input points and failed to learn anything. That run predated the architecture fix. The sweep skipped the configuration because the checkpoint file already existed, so no fresh run was ever made, and the stale broken run ended up being selected as the answer for one seed in that cell.
>
> The `n_params` audit is what makes this category of contamination reliably detectable: a 32-base-channel U-Net has a different parameter count from the 15-base-channel version, and a single-valued check on n_params per model catches it immediately. The stale run had 149641 parameters, same as the current model, which initially seemed reassuring. In fact it was coincidence — the 32-base-channel two-level architecture happened to have a similar parameter count. The timestamp-based check on `base_channels` in the `diagnose_duplicates` output was what actually identified it, and the lesson is that `n_params` alone is necessary but not sufficient: it catches count differences but not architectural changes that happen to be parameter-neutral.


### What the curves measure and why the gap matters
Each point on the error-versus-N curve represents the expected test relative L2 error for a model trained on N samples. Relative L2 error for a prediction `u_pred` against a ground truth `u_true` is the L2 norm of their difference divided by the L2 norm of `u_true`. This is the standard metric in operator learning because it normalizes for the scale of the function being predicted, making errors comparable across functions with different amplitudes.

<img width="800" height="444" alt="image" src="https://github.com/user-attachments/assets/1fa04e29-0d03-4f75-8cea-6d4e0b64af4e" />

Each cell in the grid involves five independent training runs with different random seeds, producing five error values. The seed controls both the model initialization and the batch shuffling order throughout training. Crucially, every seed at a given N trains on the same N samples — the first N rows of the training array. This means the variance bands measure optimization variance (sensitivity to initialization and batch ordering) rather than sampling variance (sensitivity to which N functions were drawn). This distinction matters for how the results are reported: the bands do not answer the question "how much would error vary if I drew a fresh N-sample dataset." They answer the question "how much does error vary across different random initializations and batch orders on this fixed dataset."

The N values are nested subsets of one another: the 100-sample training set is the first 100 rows of the 250-sample set, which is the first 250 rows of the 500-sample set, and so on. This creates correlations across the points on each curve. Because both models see identical data at every N, the correlation structure is the same for both, and the ratio between the curves — which is what the multiplier measures — is largely protected from this effect. But the individual curves are smoother than they would be under independent data draws.

The sample-complexity multiplier answers the question: at a fixed target error level, how many training samples does each model need to reach it? The ratio of those two N values is the multiplier. This is a horizontal reading of the error-versus-N plot — fixing a row (an error level) and asking at what column (an N value) each model's curve crosses it — as opposed to a vertical reading, which would fix a column and ask the error ratio at that N. The horizontal reading is the right one for a study about data efficiency, because the thesis is about data as the scarce resource. A horizontal reading is denominated in training samples, which is what the study is about.

The local power law approximation underlying the interpolation assumes that within each adjacent pair of grid points, the relationship between error and N is approximately error equals A times N to the power negative alpha for some constant alpha. Empirically, alpha is not constant: for FNO it ranges from 0.73 at the N=100 to N=250 interval up to 0.85 at N=500 to N=1000, then declines to 0.66 at the high end. For U-Net it rises monotonically from 0.39 at the low end to 0.95 at N=1500 to N=2000. This non-constancy means two things. First, quoted N estimates from interpolation should carry limited precision — approximately 300 and approximately 1250. Second, the trend itself is interpretable: FNO's alpha is decelerating, suggesting it is approaching an error floor set by something other than data quantity, while U-Net's is accelerating, suggesting it is still in a regime where more data helps substantially.

The mechanistic explanation for FNO's deceleration is worth discussing more precisely. Viscous Burgers at viscosity 0.01 on a unit domain develops near-shocks with thickness approximately 2 times viscosity divided by the velocity jump, roughly 0.02. Resolving a feature of that width requires Fourier wavenumbers up to approximately 50. The FNO's spectral truncation retains only the lowest 16 modes. This means the FNO structurally cannot represent the shock interior — the modes required to do so are zeroed before the inverse FFT. As N grows and data is no longer the binding constraint, this architectural limitation becomes the floor. The U-Net has no such floor — it can represent any continuous function given sufficient depth and width — so its error keeps declining as data accumulates, closing the gap at high N. This explains why the FNO advantage ratio peaks around N=1000 and narrows at both ends of the grid.

> One test of this hypothesis requires no additional training: take the test ground-truth solutions, apply an `rfft`, zero all modes at index 16 and above, and apply an `irfft`. The resulting reconstruction error is the best possible error achievable by any model whose output lives in the span of the lowest 16 Fourier modes. If this projection error is close to the FNO's plateau at N=2000, it supports the truncation floor hypothesis. If it is substantially lower, the FNO's pointwise and nonlinear branches are contributing significantly to high-frequency reconstruction and the story is more complicated. 

### Results

* **Headline**: to reach 10% test relative L2 error, the matched-budget U-Net requires approximately 4.1 times the training data of the FNO. The 95% confidence interval from a 5-seed bootstrap is 3.9 to 4.3. This figure comes from interpolated crossing points of approximately N=300 for the FNO and approximately N=1250 for the U-Net, on a six-point grid of N in 100, 250, 500, 1000, 1500, 2000, five seeds each, 500 training epochs per run.

<img width="800" height="572" alt="image" src="https://github.com/user-attachments/assets/fc7eafda-c773-447a-b827-eae80f74954a" />

The multiplier is stable across target error levels. Evaluated at 20% it is approximately 4.2, at 10% it is 4.09, and at 7% it is approximately 3.9. 

* The FNO has 139,745 parameters and the U-Net has 149,641 — the U-Net has 7% more parameters and still requires 4 times the data. Both parameter counts are reported as-is throughout, without rounding to a shared approximate value.

|model|	N|	test mean|	test std|	train mean|	gap|
|------|---|---------|---------|--------------|----|
|fno	|100|	0.2304	|0.0131|	0.0311|	0.199|
|fno|	250|	0.1183|	0.0078	|0.0227|	0.096|
|fno|	500|	0.0669|	0.0042|	0.0163|	0.051|
|fno|	1000|	0.0370|	0.0021|	0.0126|	0.024|
|fno|	1500|	0.0278|	0.0020|	0.0115|	0.016|
|fno	|2000|	0.0230	|0.0010|	0.0109|	0.012|
|unet|	100|	0.3957|	0.0090|	0.0350|	0.361|
|unet|	250	|0.3074	|0.0075|	0.0314|	0.276|
|unet|	500|	0.2015|	0.0044|	0.0290|	0.172|
|unet|	1000|	0.1241|	0.0068|	0.0259|	0.098|
|unet|	1500|	0.0843	|0.0034|	0.0238|	0.060|
|unet	|2000|	0.0656|	0.0046|	0.0224|	0.043|

> At N=2000, FNO's train-to-test generalization gap is 0.012 and U-Net's is 0.043, a ratio of 3.6 on identical data at matched budget. This is the same inductive-bias result read along a different axis — not how much data each model needs but how well each model generalizes from the same amount of data. 

* Both models fit their training data well at every N. Train error ranges from 0.011 to 0.035 across all cells. Neither model underfits anywhere, including at N=100. At the U-Net's crossing point near N=1500, train error is 0.0238 and test error is 0.0843. The model fits the training set fine and fails to generalize. That is data starvation, not budget starvation. This eliminates the concern that the sweep's fixed training budget was handicapping the U-Net rather than measuring its true data requirements.




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
