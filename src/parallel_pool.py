from concurrent.futures import ProcessPoolExecutor
from numpy.random import SeedSequence
from .data_gen import generate_call_data
from .sim_core import simulate_call_center


def _simulate_one(args):
    """Run one independent call-center simulation."""
    seed, n_calls, arrival_rate, service_time, n_agents, target_wait = args

    calls = generate_call_data(
        n_calls=n_calls,
        arrival_rate_per_minute=arrival_rate,
        mean_service_time_seconds=service_time,
        seed=seed,
    )

    summary, _ = simulate_call_center(
        calls,
        n_agents=n_agents,
        target_wait_sec=target_wait,
    )

    return (
        summary["n_calls"],
        summary["answered_within_target"],
        summary["average_wait_sec"] * summary["n_calls"],
    )


def run_parallel(
    n_replications=20,
    n_calls=1000,
    n_agents=9,
    arrival_rate=2.0,
    service_time=180,
    workers=1,
    seed=42,
    target_wait=30,
):
    """Run identical simulation work with different worker counts."""

    if n_replications < 1 or n_calls < 1 or workers < 1:
        raise ValueError("Replications, calls and workers must be positive")

    # Seeds are independent of how many workers we use.
    child_seeds = SeedSequence(seed).spawn(n_replications)

    tasks = [
        (
            int(child.generate_state(1)[0]),
            n_calls,
            arrival_rate,
            service_time,
            n_agents,
            target_wait,
        )
        for child in child_seeds
    ]

    if workers == 1:
        # Actual sequential execution, without pool overhead.
        results = list(map(_simulate_one, tasks))
    else:
        # Independent replications are distributed among CPU processes.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(
                _simulate_one,
                tasks,
                chunksize=max(1, n_replications // (workers * 4)),
            ))

    total_calls = sum(result[0] for result in results)
    total_answered = sum(result[1] for result in results)
    total_wait = sum(result[2] for result in results)

    return {
        "workers": workers,
        "n_replications": n_replications,
        "n_calls": total_calls,
        "n_agents": n_agents,
        "answered_within_target": total_answered,
        "service_level": total_answered / total_calls,
        "average_wait_sec": total_wait / total_calls,
    }
