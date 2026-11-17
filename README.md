Call-Center Staffing Simulation
Project overview
This project simulates a multi-agent call centre and evaluates parallel computing approaches for running large numbers of independent simulation replications. The operational question is:
How many call-centre agents are needed per shift to answer at least 80% of calls within 30 seconds, using the fewest agents possible?

The model uses synthetic call arrivals and service times so that the simulation can be reproduced with fixed random seeds. The repository currently includes a sequential reference implementation, a process-pool implementation, a NumPy-batched implementation, correctness tests, and strong- and weak-scaling benchmark results.
Model and assumptions
The current generator and simulation implement a constant-rate, multi-server queue with the following assumptions:
- Call arrivals: inter-arrival times are exponentially distributed. This produces a Poisson arrival process with a constant arrival rate.
- Service times: service durations are exponentially distributed with a configurable mean.
- Queue discipline: calls are processed in first-in, first-out (FIFO) arrival order.
- Agents: all agents are treated as identical servers. An arriving call starts service immediately when an agent is free; otherwise it waits for the earliest available agent.
- Service-level target: a call is counted as answered within the target when its waiting time is less than or equal to 30 seconds (or another configured threshold).
The default benchmark parameters are 2 calls per minute, a mean service time of 180 seconds, 9 agents, and a 30-second waiting-time target. The current benchmark experiments hold the staffing level at 9 agents so that the parallel implementations can be compared under the same workload.

The current repository snapshot does not yet implement hourly-varying arrival rates, customer abandonment, or agent breaks and shift changes.
Repository structure
.
├── README.md
├── requirements.txt
├── run_scaling.sbatch
├── runs/
│   └── benchmark.py
├── src/
│   ├── baseline.py
│   ├── data_gen.py
│   ├── metrics.py
│   ├── parallel_numpy.py
│   ├── parallel_pool.py
│   └── sim_core.py
├── tests/
│   └── test_correctness.py
└── results/
    ├── csv/
    ├── logs/
    └── plots/
Main modules
File	Purpose
src/data_gen.py	Generates reproducible synthetic arrival and service-time data and can save it as CSV.
src/sim_core.py	Implements the event-by-event call-centre simulation, per-call records, service-level summary, and a search for the minimum agent count on a supplied call dataset.
src/baseline.py	Runs the sequential simulation and the minimum-agent search.
src/parallel_pool.py	Distributes independent simulation replications across worker processes using ProcessPoolExecutor.
src/parallel_numpy.py	Processes batches of independent replications using NumPy arrays; batches can be distributed across threads.
runs/benchmark.py	Measures runtime and computes strong- and weak-scaling statistics for the process-pool implementation.
tests/test_correctness.py	Checks data generation, the sequential simulation, parallel equivalence, and worker-count invariance.
run_scaling.sbatch	Submits the strong- and weak-scaling benchmark jobs to the cluster using Slurm.


Parallel approaches
1. Sequential reference
The sequential simulation processes calls in arrival order. A min-heap stores the time at which each agent becomes available, allowing the simulation to select the next available agent efficiently. This implementation provides the reference behaviour used in correctness comparisons.
2. Process pool
The process-pool implementation assigns independent simulation replications to separate worker processes. Since replications do not depend on one another, they can be executed concurrently. NumPy's SeedSequence.spawn creates a deterministic child seed for each replication, so changing the number of workers does not change the random stream assigned to a replication.
3. NumPy batching
The NumPy implementation stores the state of multiple independent replications in arrays. At each call index, it updates the corresponding calls across the batch together. Events within any individual replication remain sequential because the next event depends on the agents' previous availability. A ThreadPoolExecutor can distribute batches across threads.
Installation and execution
Run commands from the repository root (the directory containing src/ and runs/).
Install dependencies
python -m pip install -r requirements.txt
Generate sample data
python -m src.data_gen
This creates results/csv/calls.csv using the sample settings in src/data_gen.py.
Run the sequential baseline
python -m src.baseline
The baseline prints the results for its configured staffing level and searches the supplied sample dataset for the smallest agent count that reaches the configured service-level target.
Run correctness tests
python -m pytest -q
Run a local scaling benchmark
python runs/benchmark.py --mode both --workers 1 2 4 8 --replications 64 --replications-per-worker 8 --calls 20000 --reps 5
The benchmark writes raw and summary CSV files under results/csv/ and scaling plots under results/plots/. Use a distinct --tag value to avoid overwriting a previous experiment's output files.
Correctness and reproducibility
The tests check that generated call data can be saved and loaded without changing its values, that the sequential simulation returns consistent call records and summary counts, and that the process-pool and NumPy implementations agree with a sequential reference on the same replication seeds. The tests also check that changing the worker count does not change the aggregated results.

Call counts and the number answered within the target are compared exactly. Floating-point summary measures are compared with tight numerical tolerances.

The current tests do not yet implement the planned analytical comparison with the Erlang C model. That validation should be added before describing the analytical check as completed.
Cluster benchmark results
The committed cluster results cover worker counts of 1, 2, 4, 8, 16, and 32. Each worker-count configuration was timed five times. The summaries below report the median runtime across those five repetitions.
Strong scaling
Strong scaling keeps the total workload fixed at 2,400 replications, with 300,000 calls per replication (720 million simulated calls per timed run).
Workers	Median runtime (s)	Speedup	Parallel efficiency
1	647.178770	1.0000×	100.00%
2	322.607647	2.0061×	100.30%
4	168.595766	3.8386×	95.97%
8	92.601592	6.9889×	87.36%
16	62.151774	10.4129×	65.08%
32	45.125548	14.3417×	44.82%


With 32 workers, the measured median runtime falls from approximately 647 seconds to 45 seconds, corresponding to a 14.34× speedup. Parallel efficiency decreases as the worker count grows, reaching approximately 44.82% at 32 workers. The small efficiency value above 100% at 2 workers is a measurement effect and should not be interpreted as a general superlinear-scaling result.

Weak scaling
Weak scaling increases the number of replications in proportion to the number of workers. Each worker is assigned 60 replications, and each replication contains 300,000 calls.
Workers	Total replications	Median runtime (s)	Scaled speedup	Parallel efficiency
1	60	17.633853	1.0000×	100.00%
2	120	17.762197	1.9855×	99.28%
4	240	17.813544	3.9597×	98.99%
8	480	19.150534	7.3664×	92.08%
16	960	22.860347	12.3420×	77.14%
32	1,920	35.804321	15.7602×	49.25%


Runtime remains close to constant through 4 workers, then increases as the workload and worker count grow further. At 32 workers, the median runtime is approximately 35.80 seconds and efficiency is 49.25%, indicating that the largest configuration does not maintain ideal weak scaling.

Benchmark environment
The supplied Slurm output records the cluster host as haskell and the processor model as AMD EPYC 9755. The batch script requested 40 CPUs per task. The benchmark uses five timed repetitions per worker count and performs an untimed warm-up before timing.

The raw measurements, summary tables, and Slurm output are retained in results/csv/ and results/logs/ for inspection and reproducibility.
Current scope and next steps
The current repository snapshot contains the sequential, process-pool, and NumPy-batched implementations and their benchmark outputs. The following items remain as extensions or validation work:
- Add the analytical Erlang C calculation and compare the simulated steady-state service level against it, with an explicit sampling tolerance and warm-up treatment.
- Extend the arrival generator to represent demand that varies by hour, then evaluate staffing by shift or time period.
- Implement and benchmark the proposed Cython version if it remains part of the team's claimed techniques.
- Complete src/metrics.py if a shared metrics module is still required by the final design.
- Run the staffing search across sufficiently large, representative call datasets and report the minimum-agent result separately from the fixed-staffing parallel benchmarks.
These items should only be marked as completed once their implementations and results have been added to the repository.
