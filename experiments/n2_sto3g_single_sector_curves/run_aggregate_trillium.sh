#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=n2_sto3g_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=n2_sto3g_aggregate_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davood.dar@utoronto.ca

set -euo pipefail

module load scipy-stack/2026a

slurm_account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$slurm_account/$USER/quasisymmetry}"
experiment_dir="$project_dir/experiments/n2_sto3g_single_sector_curves"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
single_sector_dir="${SINGLE_SECTOR_OO_DIR:-$HOME/links/projects/$slurm_account/$USER/single_sector_oo}"
run_dir="${N2_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/n2/sto-3g/single_sector_family_curves_20260801}"

export SINGLE_SECTOR_OO_DIR="$single_sector_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$run_dir/.matplotlib/aggregate"
export XDG_CACHE_HOME="$run_dir/.cache/aggregate"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Aggregating: $run_dir"

srun --ntasks=1 --cpus-per-task=1 \
    "$venv_dir/bin/python" -u \
    "$experiment_dir/run_n2_sto3g_single_sector_family_curves.py" \
    --output-dir "$run_dir" \
    --fixed-sector-branches 4 \
    --no-maxiter \
    --aggregate-only

echo "Finished: $(date --iso-8601=seconds)"
