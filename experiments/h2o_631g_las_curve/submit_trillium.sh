#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
account="${SLURM_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$(cd "$script_dir/../.." && pwd)}"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
parent_root="${H2O_631G_PARENT_CURVE_RUN_DIR:?H2O_631G_PARENT_CURVE_RUN_DIR is required}"
run_dir="${H2O_631G_LAS_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/h2o/6-31g/las_curve_$(date +%Y%m%d_%H%M%S)}"
array_spec="${H2O_631G_LAS_ARRAY_SPEC:-0-12%13}"

mkdir -p "$run_dir/logs"
export_values="ALL,LAS_PROJECT_DIR=$project_dir,LAS_VENV=$venv_dir,H2O_631G_PARENT_CURVE_RUN_DIR=$parent_root,H2O_631G_LAS_CURVE_RUN_DIR=$run_dir"
if [[ "${H2O_631G_SKIP_CASSCF:-0}" == "1" ]]; then
    export_values+=",H2O_631G_SKIP_CASSCF=1"
fi

fixed_job="$(sbatch --parsable \
    --account="$account" \
    --export="$export_values" \
    --output="$run_dir/logs/fixed_%j.out" \
    "$script_dir/prepare_fixed_trillium.sh")"

array_job="$(sbatch --parsable \
    --account="$account" \
    --dependency="afterok:$fixed_job" \
    --array="$array_spec" \
    --export="$export_values" \
    --output="$run_dir/logs/array_%A_%a.out" \
    "$script_dir/run_task_trillium.sh")"

aggregate_job="$(sbatch --parsable \
    --account="$account" \
    --dependency="afterok:$array_job" \
    --export="$export_values" \
    --output="$run_dir/logs/aggregate_%j.out" \
    "$script_dir/aggregate_trillium.sh")"

echo "Fixed equilibrium-selection job: $fixed_job"
echo "Geometry array: $array_job ($array_spec)"
echo "Aggregation job: $aggregate_job"
echo "Run directory: $run_dir"
echo "Monitor: squeue -j $fixed_job,$array_job,$aggregate_job"
