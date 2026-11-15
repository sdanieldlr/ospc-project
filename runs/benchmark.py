"""Benchmark: runtime, speedup, efficiency and scaling of the process-pool version.

Run from the project root (the folder that contains src/ and runs/):

    python runs/benchmark.py --mode strong --workers 1 2 4 8 --replications 64 --calls 20000 --reps 5
    python runs/benchmark.py --mode weak   --workers 1 2 4 8 --replications-per-worker 8 --calls 20000 --reps 5

Outputs (created automatically):
    results/csv/<mode>_raw.csv       one row per timed repetition
    results/csv/<mode>_summary.csv   median / min / max per worker count + speedup + efficiency
    results/plots/<mode>_scaling.png plots for the report and slides
"""

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

# Make "import src..." work when the file is run as `python runs/benchmark.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from numpy.random import SeedSequence  # noqa: E402

from src.parallel_pool import _simulate_one, run_parallel  # noqa: E402

CSV_DIR = ROOT / "results" / "csv"
PLOT_DIR = ROOT / "results" / "plots"


def run_sequential(n_replications, n_calls, n_agents, arrival_rate, service_time, seed, target_wait):
    """Same work as run_parallel, but in one process with no pool: the true sequential reference."""
    child_seeds = SeedSequence(seed).spawn(n_replications)
    total_calls = total_answered = 0
    total_wait = 0.0
    for child in child_seeds:
        task = (int(child.generate_state(1)[0]), n_calls, arrival_rate, service_time, n_agents, target_wait)
        calls, answered, wait = _simulate_one(task)
        total_calls += calls
        total_answered += answered
        total_wait += wait
    return {
        "service_level": total_answered / total_calls,
        "average_wait_sec": total_wait / total_calls,
    }


def time_once(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def check_same_result(reference, other, label):
    """Correctness guard: a parallel run must give the same answer as the sequential run."""
    ok = math.isclose(reference["service_level"], other["service_level"], rel_tol=0, abs_tol=1e-12)
    ok = ok and math.isclose(reference["average_wait_sec"], other["average_wait_sec"], rel_tol=1e-9)
    if not ok:
        print(f"WARNING: {label} gives a different result from the sequential run: {other} vs {reference}")
    return ok


def benchmark(mode, workers_list, args):
    raw_rows = []
    summary = {}

    for workers in workers_list:
        if mode == "strong":
            n_rep = args.replications  # same total work for every worker count
        else:
            n_rep = args.replications_per_worker * workers  # work grows with workers

        kwargs = dict(
            n_replications=n_rep,
            n_calls=args.calls,
            n_agents=args.agents,
            arrival_rate=args.arrival_rate,
            service_time=args.service_time,
            seed=args.seed,
            target_wait=args.target_wait,
        )
        seq_kwargs = {k: v for k, v in kwargs.items()}

        # Untimed warm-up so imports and first process start-up do not distort the first repetition.
        run_parallel(workers=workers, **kwargs)

        reference = run_sequential(**seq_kwargs)
        times = []
        for rep in range(1, args.reps + 1):
            elapsed, result = time_once(lambda: run_parallel(workers=workers, **kwargs))
            check_same_result(reference, result, f"{workers} workers, rep {rep}")
            times.append(elapsed)
            raw_rows.append(
                {
                    "mode": mode,
                    "workers": workers,
                    "n_replications": n_rep,
                    "n_calls_per_replication": args.calls,
                    "repetition": rep,
                    "seconds": round(elapsed, 6),
                    "service_level": result["service_level"],
                    "average_wait_sec": result["average_wait_sec"],
                }
            )
            print(f"[{mode}] workers={workers:>3} rep={rep}/{args.reps} time={elapsed:.3f}s")

        summary[workers] = {
            "n_replications": n_rep,
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        }

    # Speedup / efficiency are computed from the MEDIAN time, relative to the smallest worker count (normally 1).
    base_workers = min(summary)
    base_time = summary[base_workers]["median"]
    summary_rows = []
    for workers in sorted(summary):
        s = summary[workers]
        if mode == "strong":
            speedup = base_time / s["median"]  # T1 / Tp
            efficiency = speedup / (workers / base_workers)  # S / p
        else:
            speedup = (base_time / s["median"]) * (workers / base_workers)  # scaled speedup
            efficiency = base_time / s["median"]  # T1 / Tp, ideal = 1
        summary_rows.append(
            {
                "mode": mode,
                "workers": workers,
                "n_replications": s["n_replications"],
                "median_s": round(s["median"], 6),
                "min_s": round(s["min"], 6),
                "max_s": round(s["max"], 6),
                "stdev_s": round(s["stdev"], 6),
                "speedup": round(speedup, 4),
                "efficiency": round(efficiency, 4),
            }
        )
    return raw_rows, summary_rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def estimate_serial_fraction(summary_rows):
    """Amdahl: S = 1 / (s + (1 - s) / p)  ->  s = (1/S - 1/p) / (1 - 1/p), using the largest worker count."""
    last = summary_rows[-1]
    p, s_meas = last["workers"], last["speedup"]
    if p <= 1 or s_meas <= 0:
        return None
    frac = (1 / s_meas - 1 / p) / (1 - 1 / p)
    return min(max(frac, 0.0), 1.0)


def make_plots(mode, summary_rows, suffix=""):
    import matplotlib

    matplotlib.use("Agg")  # no display needed (cluster)
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    workers = [r["workers"] for r in summary_rows]
    speedup = [r["speedup"] for r in summary_rows]
    efficiency = [r["efficiency"] for r in summary_rows]
    median = [r["median_s"] for r in summary_rows]
    lo = [r["median_s"] - r["min_s"] for r in summary_rows]
    hi = [r["max_s"] - r["median_s"] for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].errorbar(workers, median, yerr=[lo, hi], marker="o", capsize=4)
    axes[0].set_xlabel("workers")
    axes[0].set_ylabel("runtime (s), median with min-max")
    axes[0].set_title("Runtime")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(workers, [str(w) for w in workers])
    if mode == "weak":
        axes[0].axhline(median[0], linestyle="--", color="gray", label="ideal (constant)")
        axes[0].legend()

    axes[1].plot(workers, speedup, marker="o", label="measured")
    axes[1].plot(workers, [w / workers[0] for w in workers], linestyle="--", color="gray", label="ideal")
    serial = estimate_serial_fraction(summary_rows) if mode == "strong" else None
    if serial is not None:
        amdahl = [1 / (serial + (1 - serial) / (w / workers[0])) for w in workers]
        axes[1].plot(workers, amdahl, linestyle=":", color="tab:red", label=f"Amdahl, s={serial:.2f}")
    axes[1].set_xlabel("workers")
    axes[1].set_ylabel("speedup" if mode == "strong" else "scaled speedup")
    axes[1].set_title("Speedup")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(workers, [str(w) for w in workers])
    axes[1].legend()

    axes[2].plot(workers, efficiency, marker="o")
    axes[2].axhline(1.0, linestyle="--", color="gray")
    axes[2].set_ylim(0, 1.1)
    axes[2].set_xlabel("workers")
    axes[2].set_ylabel("efficiency")
    axes[2].set_title("Efficiency")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(workers, [str(w) for w in workers])

    fig.suptitle(f"{mode.capitalize()} scaling of the process-pool simulation")
    fig.tight_layout()
    out = PLOT_DIR / f"{mode}_scaling{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Scaling benchmark for the call-center simulation")
    parser.add_argument("--mode", choices=["strong", "weak", "both"], default="strong")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--replications", type=int, default=64, help="strong scaling: total replications")
    parser.add_argument("--replications-per-worker", type=int, default=8, help="weak scaling: replications per worker")
    parser.add_argument("--calls", type=int, default=20000, help="calls per replication")
    parser.add_argument("--reps", type=int, default=5, help="timed repetitions per worker count")
    parser.add_argument("--agents", type=int, default=9)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--service-time", type=float, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-wait", type=float, default=30)
    parser.add_argument("--tag", default="", help="optional suffix for the output files, e.g. laptop or cluster")
    args = parser.parse_args()

    modes = ["strong", "weak"] if args.mode == "both" else [args.mode]
    workers_list = sorted(set(args.workers))
    suffix = f"_{args.tag}" if args.tag else ""

    for mode in modes:
        raw_rows, summary_rows = benchmark(mode, workers_list, args)
        save_csv(CSV_DIR / f"{mode}_raw{suffix}.csv", raw_rows)
        save_csv(CSV_DIR / f"{mode}_summary{suffix}.csv", summary_rows)
        make_plots(mode, summary_rows, suffix)
        serial = estimate_serial_fraction(summary_rows) if mode == "strong" else None
        if serial is not None:
            print(f"Estimated serial fraction (Amdahl, largest worker count): {serial:.3f}")
            if serial > 0:
                print(f"Amdahl maximum speedup 1/s = {1 / serial:.1f}")


if __name__ == "__main__":
    main()
