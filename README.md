# OSPC Group Project

This project will compare a sequential simulation baseline with parallel
approaches. The team will validate that the implementations agree, then
measure runtime and scaling behavior.

## Project layout

- `src/sim_core.py` — shared simulation logic
- `src/baseline.py` — sequential reference implementation
- `src/parallel_pool.py` — process-pool approach
- `src/parallel_numpy.py` — second parallel or batched approach
- `src/data_gen.py` — reproducible synthetic inputs
- `src/metrics.py` — service-level and waiting-time metrics
- `tests/test_correctness.py` — baseline and parallel correctness checks
- `runs/benchmark.py` — runtime and scaling experiments
- `results/` — generated CSV files and plots

Implementation, usage instructions, and dependencies will be added as the
project develops.
