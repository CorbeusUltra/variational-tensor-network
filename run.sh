#!/bin/bash
#SBATCH --job-name=vtn-diagnostic
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/%j.out
#SBATCH --error=outputs/logs/%j.err

module purge
module load gcc
module load python/3.13.0

uv sync

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=4

srun /usr/bin/time -v .venv/bin/python -u scripts/su_ctmrg_convergence.py
