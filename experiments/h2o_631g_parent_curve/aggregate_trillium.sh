#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=h2o631_refsum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00

set -euo pipefail

module load scipy-stack/2026a
account="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$HOME/links/projects/$account/$USER/quasisymmetry}"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
run_dir="${H2O_631G_PARENT_CURVE_RUN_DIR:?H2O_631G_PARENT_CURVE_RUN_DIR is required}"

"$venv_dir/bin/python" -u \
    "$project_dir/experiments/h2o_631g_parent_curve/run_parent_curve.py" \
    --project-dir "$project_dir" \
    --output-dir "$run_dir" \
    --python "$venv_dir/bin/python" \
    --aggregate-only
