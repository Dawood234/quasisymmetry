#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o631_las_curve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300

set -euo pipefail

module load scipy-stack/2026a

account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$account/$USER/quasisymmetry}"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
parent_root="${H2O_631G_PARENT_CURVE_RUN_DIR:?H2O_631G_PARENT_CURVE_RUN_DIR is required}"
run_dir="${H2O_631G_LAS_CURVE_RUN_DIR:?H2O_631G_LAS_CURVE_RUN_DIR is required}"
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

arguments=(
    --project-dir "$project_dir"
    --parent-root "$parent_root"
    --output-dir "$run_dir"
    --task-index "$task_index"
    --candidate-workers 10
    --candidate-threads 192
    --dmrg-threads 32
    --casscf-threads 16
    --resume
)
if [[ "${H2O_631G_SKIP_CASSCF:-0}" == "1" ]]; then
    arguments+=(--skip-casscf)
fi

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: $project_dir"
echo "Parent root: $parent_root"
echo "LAS run: $run_dir"
echo "Array task: $task_index"

srun --ntasks=1 --cpus-per-task=192 \
    "$venv_dir/bin/python" -u \
    "$project_dir/experiments/h2o_631g_las_curve/run_curve.py" \
    "${arguments[@]}"

echo "Finished: $(date --iso-8601=seconds)"
