#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=n2_sto3g_curve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-20%21
#SBATCH --output=n2_sto3g_curve_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=davood.dar@utoronto.ca

set -euo pipefail

module load scipy-stack/2026a

slurm_account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$slurm_account/$USER/quasisymmetry}"
experiment_dir="$project_dir/experiments/n2_sto3g_single_sector_curves"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
run_dir="${N2_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/n2/sto-3g/single_sector_initial_lowest_curves_20260801}"
threads="${SLURM_CPUS_PER_TASK:-8}"

distances=(
    1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90
    2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00
)
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
distance="${distances[$task_index]}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Python environment not found: $venv_dir" >&2
    exit 2
fi
if [[ ! -f "$experiment_dir/run_n2_sto3g_single_sector_family_curves.py" ]]; then
    echo "Experiment driver not found: $experiment_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
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
echo "Array task: $task_index / 20"
echo "N-N distance: $distance Angstrom"
echo "Threads: $threads"
echo "Project: $project_dir"
echo "Backend: quasisymmetry sector utilities + ffsim"
echo "Run directory: $run_dir"

srun --ntasks=1 --cpus-per-task="$threads" \
    "$venv_dir/bin/python" -u \
    "$experiment_dir/run_n2_sto3g_single_sector_family_curves.py" \
    --grid "$distance" \
    --output-dir "$run_dir" \
    --maxiter 60 \
    --resume \
    --no-aggregate

echo "Finished: $(date --iso-8601=seconds)"
