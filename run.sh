#!/bin/bash
#SBATCH --job-name=vtn-analysis
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/logs/%j.out
#SBATCH --error=outputs/logs/%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge

export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun /usr/bin/time -v .venv/bin/python -u scripts/analysis.py
