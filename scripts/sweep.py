# scripts/sweep.py
import os, sys, subprocess, argparse, time

N_GRID = [100, 250, 500, 1000, 1500, 2000]
MODELS = ["fno", "unet"]
SEEDS  = [0, 1, 2, 3, 4]

LOG_DIR = "logs/sweep"

def run_one(model_type, n_train, seed, dry_run=False, force=False):
    ckpt = f"checkpoints/{model_type}_N{n_train}_seed{seed}_final.eqx"
    if os.path.exists(ckpt) and not force:
        print(f"[skip] {ckpt} already exists")
        return "skipped"

    cmd = [
        sys.executable, "-m", "scripts.train_model",
        "--model", model_type,
        "--n_train", str(n_train),
        "--seed", str(seed),
    ]

    if dry_run:
        print(f"[dry_run] {' '.join(cmd)}")
        return "dry_run"

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/{model_type}_N{n_train}_seed{seed}.log"

    print(f"[run] {model_type} N={n_train} seed={seed} -> {log_path}")
    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        # subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT) launches train_model.py as a completely separate Python process — fresh interpreter, fresh JAX/XLA backend, no shared JIT cache with the sweep script or with any prior run. stderr=subprocess.STDOUT merges error output into the same log file as stdout, so a traceback lands right after the last printed epoch, not in a separate stream you'd have to go hunting for.
    dt = time.time() - t0

    if result.returncode == 0:
        # result.returncode is the subprocess's exit code — 0 means the script ran to completion, nonzero means it crashed (uncaught exception, OOM, etc.). That's how "log and continue" is implemented: we check the code, print a one-line failure notice, but never raise, so the outer loop keeps going.
        print(f"  done in {dt:.0f}s")
        return "success"
    else:
        print(f"  FAILED (exit {result.returncode}) — see {log_path}")
        return "failed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the planned commands without running them.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run configs even if a checkpoint exists.")
    args = parser.parse_args()

    results = {"success": [], "failed": [], "skipped": [], "dry_run": []}

    for n_train in N_GRID:
        for model_type in MODELS:
            for seed in SEEDS:
                status = run_one(model_type, n_train, seed, dry_run=args.dry_run, force=args.force)
                results[status].append((model_type, n_train, seed))

total = len(results["success"]) + len(results["failed"]) + len(results["skipped"])
print("\n" + "=" * 50)
print(f"sweep complete: {total} configs")
print(f"  success: {len(results['success'])}")
print(f"  skipped: {len(results['skipped'])}")
print(f"  failed:  {len(results['failed'])}")
if results["failed"]:
    print("\nfailed configs:")
    for model_type, n_train, seed in results["failed"]:
        print(f"  {model_type} N={n_train} seed={seed} — see {LOG_DIR}/{model_type}_N{n_train}_seed{seed}.log")