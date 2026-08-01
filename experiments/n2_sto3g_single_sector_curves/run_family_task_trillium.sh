#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=n2_sto3g_fam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Trillium assigns the complete node memory automatically.
#SBATCH --time=23:00:00
#SBATCH --array=0-3%4
#SBATCH --output=n2_sto3g_family_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=davood.dar@utoronto.ca

set -euo pipefail

module load scipy-stack/2026a

slurm_account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$slurm_account/$USER/quasisymmetry}"
experiment_dir="$project_dir/experiments/n2_sto3g_single_sector_curves"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
run_dir="${N2_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/n2/sto-3g/single_sector_continuation_curves_20260801}"
threads="${SLURM_CPUS_PER_TASK:-8}"

families=(all-even seniority cas quartet)
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
family="${families[$task_index]}"

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
export MPLCONFIGDIR="$run_dir/.matplotlib/family_$task_index"
export XDG_CACHE_HOME="$run_dir/.cache/family_$task_index"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Array task: $task_index / 3"
echo "Family: $family"
echo "Project: $project_dir"
echo "Experiment: $experiment_dir"
echo "Run directory: $run_dir"
echo "Threads: $threads"
echo "Mode: ordered geometry continuation within one family"

srun --ntasks=1 --cpus-per-task="$threads" \
    "$venv_dir/bin/python" -u \
    "$experiment_dir/run_n2_sto3g_single_sector_family_curves.py" \
    --families "$family" \
    --output-dir "$run_dir" \
    --maxiter 60 \
    --resume \
    --no-aggregate

echo "Finished: $(date --iso-8601=seconds)"
