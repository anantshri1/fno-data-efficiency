# scripts/plot_results.py
import mlflow
import pandas as pd
import numpy as np

from configs.default import Config

N_GRID = [100, 250, 500, 1000, 1500, 2000]
SEEDS  = [0, 1, 2, 3, 4]

def load_sweep_results() -> pd.DataFrame:
    cfg = Config()
    mlflow.set_experiment(cfg.experiment_name)

    runs = mlflow.search_runs(experiment_names=[cfg.experiment_name])

    runs["n_train"]    = runs["params.n_train"].astype(int)
    runs["seed"]       = runs["params.seed"].astype(int)
    runs["model_type"] = runs["params.model_type"]

    runs = runs[
        runs["model_type"].isin(["fno", "unet"]) &   # <-- excludes legacy/untyped runs
        runs["n_train"].isin(N_GRID) &
        runs["seed"].isin(SEEDS)
    ]

    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    # metrics.test_rel_l2 in search_runs() is each run's *last logged value*
    # for that metric — i.e. exactly the final-epoch test rel-L2, since
    # train_model.py logs it every epoch via mlflow.log_metrics(step=epoch).
    summary = (
        runs.groupby(["model_type", "n_train"])
        .agg(
            mean=("metrics.test_rel_l2", "mean"),
            std=("metrics.test_rel_l2", "std"),
            train_mean=("metrics.train_rel_l2", "mean"),
            train_std=("metrics.train_rel_l2", "std"),
            n_seeds=("metrics.test_rel_l2", "count"),
        )
        .reset_index()
        .sort_values(["model_type", "n_train"])
    )
    summary["gap"] = summary["mean"] - summary["train_mean"]
    return summary

def diagnose_duplicates(runs: pd.DataFrame):
    dupe_groups = runs.groupby(["model_type", "n_train", "seed"]).size()
    dupes = dupe_groups[dupe_groups > 1]
    print(f"\n{len(dupes)} (model, n_train, seed) combos have duplicate runs:\n")

    cols = ["model_type", "n_train", "seed", "start_time",
            "metrics.test_rel_l2", "params.unet_base_channels", "status"]
    cols = [c for c in cols if c in runs.columns]

    for (model_type, n_train, seed), _ in dupes.items():
        subset = runs[
            (runs["model_type"] == model_type) &
            (runs["n_train"] == n_train) &
            (runs["seed"] == seed)
        ][cols].sort_values("start_time")
        print(subset.to_string(index=False))
        print()

def inspect_cell(runs: pd.DataFrame, model_type: str, n_train: int):
    sub = runs[
        (runs["model_type"] == model_type) & (runs["n_train"] == n_train)
    ].sort_values("seed")
    cols = ["seed", "metrics.train_rel_l2", "metrics.test_rel_l2"]
    print(f"\nper-seed — {model_type} N={n_train}:")
    print(sub[cols].to_string(index=False))

def dedupe_latest(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the most recent run per (model_type, n_train, seed).
    Two sources of duplicates get collapsed here: harmless re-runs of
    the same config (deterministic training -> identical metrics, safe
    to pick either), and stale pre-fix runs (e.g. the 2-level U-Net
    era) whose metrics reflect a different, broken architecture and
    must not be averaged in. Most-recent-by-start_time is correct for
    both, since the sweep grid always postdates exploratory runs.
    """
    before = len(runs)
    runs = runs.sort_values("start_time").drop_duplicates(
        subset=["model_type", "n_train", "seed"], keep="last"
    )
    print(f"deduped {before} -> {len(runs)} runs (kept most recent per config)")
    return runs

def audit_provenance(runs: pd.DataFrame):
    print("\ndistinct n_params per model:")
    print(runs.groupby("model_type")["params.n_params"].unique())

    print("\nall runs by start_time:")
    cols = ["model_type", "n_train", "seed", "start_time",
            "params.n_params", "metrics.test_rel_l2"]
    print(runs[cols].sort_values("start_time").to_string(index=False))


import matplotlib.pyplot as plt
import os

def plot_error_vs_n(summary: pd.DataFrame, out_path: str = "results/error_vs_n.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"fno": "tab:blue", "unet": "tab:orange"}
    labels = {"fno": "FNO", "unet": "U-Net"}

    for model_type in ["fno", "unet"]:
        sub = summary[summary["model_type"] == model_type].sort_values("n_train")
        n = sub["n_train"].values
        mean = sub["mean"].values
        std = sub["std"].values

        ax.plot(n, mean, marker="o", color=colors[model_type], label=labels[model_type])
        ax.fill_between(n, mean - std, mean + std, color=colors[model_type], alpha=0.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size (N)")
    ax.set_ylabel("Test relative L2 error")
    ax.set_title("Data efficiency: FNO vs. U-Net on 1D Burgers'")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot: {out_path}")
    plt.close(fig)

def n_for_target_error(summary: pd.DataFrame, model_type: str, target_error: float) -> float:
    """
    Log-log linear interpolation: at what N does this model's mean test
    error cross target_error? Refuses to extrapolate — target_error must
    lie strictly within the model's observed error range on the grid,
    or the estimate is unsupported by data.
    """
    sub = summary[summary["model_type"] == model_type].sort_values("n_train")
    n, err = sub["n_train"].values, sub["mean"].values  # err is decreasing in n

    if target_error > err[0] or target_error < err[-1]:
        raise ValueError(
            f"{model_type}: target_error={target_error} outside observed range "
            f"[{err[-1]:.4f}, {err[0]:.4f}] — would require extrapolation."
        )

    for i in range(len(n) - 1):
        if err[i] >= target_error >= err[i + 1]:
            log_n0, log_n1 = np.log(n[i]), np.log(n[i + 1])
            log_e0, log_e1 = np.log(err[i]), np.log(err[i + 1])
            frac = (np.log(target_error) - log_e0) / (log_e1 - log_e0)
            return float(np.exp(log_n0 + frac * (log_n1 - log_n0)))

    raise RuntimeError("unreachable if monotonic and bounds check passed")

def sample_complexity_multiplier(summary: pd.DataFrame, target_error: float = 0.10):
    n_fno  = n_for_target_error(summary, "fno",  target_error)
    n_unet = n_for_target_error(summary, "unet", target_error)
    mult = n_unet / n_fno
    print(f"\nAt target test rel-L2 = {target_error:.2%}:")
    print(f"  FNO  needs  N ≈ {n_fno:6.0f}")
    print(f"  UNet needs  N ≈ {n_unet:6.0f}")
    print(f"  -> UNet requires {mult:.2f}x the training data of FNO")
    return mult

def bootstrap_multiplier(runs: pd.DataFrame, target_error: float = 0.10,
                         n_boot: int = 2000, rng_seed: int = 0):
    """
    Percentile bootstrap CI on the sample-complexity multiplier.
    Resamples the 5 seeds within each (model, N) cell with replacement,
    rebuilds the mean curve, and re-derives the crossing points. This
    propagates seed variance through the interpolation rather than
    treating the cell means as exact.
    """
    rng = np.random.default_rng(rng_seed)

    cells = {
        (mt, n): grp["metrics.test_rel_l2"].values
        for (mt, n), grp in runs.groupby(["model_type", "n_train"])
    }

    mults, n_skipped = [], 0
    for _ in range(n_boot):
        rows = [
            {"model_type": mt, "n_train": n,
             "mean": rng.choice(vals, size=len(vals), replace=True).mean()}
            for (mt, n), vals in cells.items()
        ]
        boot = pd.DataFrame(rows)
        try:
            mults.append(
                n_for_target_error(boot, "unet", target_error)
                / n_for_target_error(boot, "fno", target_error)
            )
        except (ValueError, RuntimeError):
            # resampled curve was non-monotonic or didn't bracket the target
            n_skipped += 1

    mults = np.array(mults)
    lo, hi = np.percentile(mults, [2.5, 97.5])
    print(f"\nbootstrap ({len(mults)} valid of {n_boot}, {n_skipped} skipped)")
    print(f"  multiplier at {target_error:.0%}: {mults.mean():.2f} "
          f"[{lo:.2f}, {hi:.2f}] (95% CI)")
    return mults


if __name__ == "__main__":
    runs = load_sweep_results()
    print(f"pulled {len(runs)} runs from MLflow")
    diagnose_duplicates(runs)
    runs = dedupe_latest(runs)
    audit_provenance(runs)

    missing_key = runs[runs["model_type"].isna() | runs["n_train"].isna()]
    if len(missing_key):
        print(f"\n{len(missing_key)} row(s) with missing groupby keys:")
        print(missing_key[["model_type", "n_train", "seed", "start_time",
                            "metrics.test_rel_l2", "status"]].to_string(index=False))

    summary = summarize(runs)
    print(summary.to_string(index=False))

    inspect_cell(runs, "unet", 100)
    inspect_cell(runs, "fno",  100)      # FNO N=100 as the reference cell

    bad = summary[summary["n_seeds"] != len(SEEDS)]
    if len(bad):
        print("\n WARNING — incomplete cells:")
        print(bad.to_string(index=False))
    else:
        print(f"\nall {len(summary)} cells have {len(SEEDS)}/{len(SEEDS)} seeds — clean.")

    summary.to_csv("results/sweep_summary.csv", index=False)
    print("saved summary: results/sweep_summary.csv")

    plot_error_vs_n(summary)
    sample_complexity_multiplier(summary, target_error=0.10)
    bootstrap_multiplier(runs, target_error=0.10)