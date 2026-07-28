from dataclasses import dataclass

@dataclass
class Config:
    # --- Data ---
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

    # --- Experiment ---
    data_dir: str = "data"
    experiment_name: str = "fno-data-efficiency"


