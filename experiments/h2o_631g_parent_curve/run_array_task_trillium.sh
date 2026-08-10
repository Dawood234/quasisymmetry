#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o631_parent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300

set -euo pipefail

module load scipy-stack/2026a

account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$account/$USER/quasisymmetry}"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
run_dir="${H2O_631G_PARENT_CURVE_RUN_DIR:?H2O_631G_PARENT_CURVE_RUN_DIR is required}"
threads="${SLURM_CPUS_PER_TASK:-32}"
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
working_root="${SLURM_TMPDIR:-$run_dir/local_work}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

arguments=(
    --project-dir "$project_dir"
    --output-dir "$run_dir"
    --python "$venv_dir/bin/python"
    --task-index "$task_index"
    --threads "$threads"
    --working-root "$working_root"
    --resume
)
if [[ -n "${H2O_631G_EQUILIBRIUM_REFERENCE_DIR:-}" ]]; then
    arguments+=(--reuse-equilibrium-dir "$H2O_631G_EQUILIBRIUM_REFERENCE_DIR")
fi

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: $project_dir"
echo "Run directory: $run_dir"
echo "Array task: $task_index"
echo "Threads: $threads"
echo "Working root: $working_root"
echo "Python: $venv_dir/bin/python"

srun --ntasks=1 --cpus-per-task="$threads" \
    "$venv_dir/bin/python" -u \
    "$project_dir/experiments/h2o_631g_parent_curve/run_parent_curve.py" \
    "${arguments[@]}"

echo "Finished: $(date --iso-8601=seconds)"
