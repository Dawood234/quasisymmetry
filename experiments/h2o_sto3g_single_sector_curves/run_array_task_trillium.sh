#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o_sto3g_curve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-12%13
#SBATCH --output=h2o_sto3g_curve_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=davood.dar@utoronto.ca

set -euo pipefail

module load scipy-stack/2026a

slurm_account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$slurm_account/$USER/quasisymmetry}"
experiment_dir="$project_dir/experiments/h2o_sto3g_single_sector_curves"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
single_sector_dir="${SINGLE_SECTOR_OO_DIR:-$HOME/links/projects/$slurm_account/$USER/single_sector_oo}"
run_dir="${H2O_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/h2o/sto-3g/single_sector_initial_lowest_curves_20260801}"
threads="${SLURM_CPUS_PER_TASK:-8}"

distances=(
    0.70 0.80 0.90 0.958 1.10 1.25 1.50 1.75 2.00 2.25
    2.50 2.75 3.00
)
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
distance="${distances[$task_index]}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Python environment not found: $venv_dir" >&2
    exit 2
fi
if [[ ! -f "$experiment_dir/run_h2o_sto3g_single_sector_family_curves.py" ]]; then
    echo "Experiment driver not found: $experiment_dir" >&2
    exit 2
fi
if [[ ! -f "$single_sector_dir/single_sector_oo/__init__.py" ]]; then
    echo "single_sector_oo project not found: $single_sector_dir" >&2
    echo "Set SINGLE_SECTOR_OO_DIR to the transferred project root." >&2
    exit 2
fi

mkdir -p "$run_dir"
export SINGLE_SECTOR_OO_DIR="$single_sector_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads"
export MPLCONFIGDIR="$run_dir/.matplotlib/task_$task_index"
export XDG_CACHE_HOME="$run_dir/.cache/task_$task_index"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Array task: $task_index / 12"
echo "O-H distance: $distance Angstrom"
echo "Threads: $threads"
echo "Project: $project_dir"
echo "single_sector_oo: $single_sector_dir"
echo "Run directory: $run_dir"

srun --ntasks=1 --cpus-per-task="$threads" \
    "$venv_dir/bin/python" -u \
    "$experiment_dir/run_h2o_sto3g_single_sector_family_curves.py" \
    --grid "$distance" \
    --output-dir "$run_dir" \
    --maxiter 60 \
    --resume \
    --no-aggregate

echo "Finished: $(date --iso-8601=seconds)"
