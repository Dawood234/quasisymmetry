#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o_sto3g_cont
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Trillium assigns the complete node memory automatically.
#SBATCH --time=23:00:00
#SBATCH --output=h2o_sto3g_continuation_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davood.dar@utoronto.ca

set -euo pipefail

module load scipy-stack/2026a

slurm_account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$slurm_account/$USER/quasisymmetry}"
experiment_dir="$project_dir/experiments/h2o_sto3g_single_sector_curves"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
single_sector_dir="${SINGLE_SECTOR_OO_DIR:-$HOME/links/projects/$slurm_account/$USER/single_sector_oo}"
run_dir="${H2O_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/h2o/sto-3g/single_sector_continuation_curves_20260801}"
threads="${SLURM_CPUS_PER_TASK:-8}"
maxiter="${H2O_MAXITER:-60}"
neighbor_passes="${H2O_NEIGHBOR_REFIT_PASSES:-1}"

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
export MPLCONFIGDIR="$run_dir/.matplotlib/sequential"
export XDG_CACHE_HOME="$run_dir/.cache/sequential"

grid_args=()
if [[ -n "${H2O_CURVE_GRID:-}" ]]; then
    grid_args=(--grid "$H2O_CURVE_GRID")
fi
refine_args=()
if [[ -n "${H2O_REFINE_STEP:-}" ]]; then
    refine_args=(--refine-step "$H2O_REFINE_STEP")
fi

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: $project_dir"
echo "Experiment: $experiment_dir"
echo "Run directory: $run_dir"
echo "Threads: $threads"
echo "Mode: sequential continuation plus neighbor envelope"
echo "Neighbor-refit passes: $neighbor_passes"
if [[ -n "${H2O_CURVE_GRID:-}" ]]; then
    echo "Grid override: $H2O_CURVE_GRID"
fi
if [[ -n "${H2O_REFINE_STEP:-}" ]]; then
    echo "Grid refinement step: $H2O_REFINE_STEP Angstrom"
fi

srun --ntasks=1 --cpus-per-task="$threads" \
    "$venv_dir/bin/python" -u \
    "$experiment_dir/run_h2o_sto3g_single_sector_family_curves.py" \
    "${grid_args[@]}" \
    "${refine_args[@]}" \
    --output-dir "$run_dir" \
    --maxiter "$maxiter" \
    --neighbor-refit \
    --neighbor-refit-passes "$neighbor_passes" \
    --resume

echo "Finished: $(date --iso-8601=seconds)"
