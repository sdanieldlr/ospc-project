"""Approach 2: NumPy vectorization across replications, batches spread over threads.

Calls inside one replication depend on each other, but replications are
independent, so each step handles call i of every replication in a batch at once.
Same seeds as parallel_pool.py -> same results.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.random import SeedSequence

from .data_gen import generate_arrival_times, generate_service_times


def _simulate_batch(seeds, n_calls, arrival_rate, service_time, n_agents, target_wait):
    # One row per replication (same data as generate_call_data with that seed)
    arrivals = np.array([generate_arrival_times(n_calls, arrival_rate, seed=s) * 60.0 for s in seeds])
    services = np.array([generate_service_times(n_calls, service_time, seed=s + 1) for s in seeds])

    rows = np.arange(len(seeds))
    agent_free = np.zeros((len(seeds), n_agents))  # time each agent becomes free
    answered = np.zeros(len(seeds), dtype=np.int64)
    total_wait = np.zeros(len(seeds))

    for i in range(n_calls):
        agent = agent_free.argmin(axis=1)                         # earliest free agent
        start = np.maximum(arrivals[:, i], agent_free[rows, agent])
        wait = start - arrivals[:, i]
        answered += wait <= target_wait
        total_wait += wait
        agent_free[rows, agent] = start + services[:, i]

    return len(seeds) * n_calls, int(answered.sum()), float(total_wait.sum())


def run_numpy(n_replications=20, n_calls=1000, n_agents=9, arrival_rate=2.0, service_time=180,
              workers=1, seed=42, target_wait=30, batch_size=32):
    if min(n_replications, n_calls, n_agents, workers, batch_size) < 1:
        raise ValueError("All sizes must be positive")

    seeds = [int(c.generate_state(1)[0]) for c in SeedSequence(seed).spawn(n_replications)]
    batches = [seeds[i:i + batch_size] for i in range(0, n_replications, batch_size)]

    def task(b):
        return _simulate_batch(b, n_calls, arrival_rate, service_time, n_agents, target_wait)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(task, batches))  # keeps batch order

    calls, answered, wait = (sum(r[k] for r in results) for k in range(3))
    return {
        "workers": workers,
        "n_replications": n_replications,
        "n_calls": calls,
        "n_agents": n_agents,
        "answered_within_target": answered,
        "service_level": answered / calls,
        "average_wait_sec": wait / calls,
    }


if __name__ == "__main__":
    import argparse
