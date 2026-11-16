#!/bin/bash
#SBATCH --job-name=ospc-scaling
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# One thread per process: all parallelism must come from our own process pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/ospc-project
source ~/venv/bin/activate

echo "host: $(hostname)  start: $(date)  cpus: $SLURM_CPUS_PER_TASK"
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Socket\(s\)"

python runs/benchmark.py --mode strong --workers 1 2 4 8 16 32 --replications 2400 --calls 300000 --reps 5 --tag cluster
python runs/benchmark.py --mode weak --workers 1 2 4 8 16 32 --replications-per-worker 60 --calls 300000 --reps 5 --tag cluster
echo "end: $(date)"
