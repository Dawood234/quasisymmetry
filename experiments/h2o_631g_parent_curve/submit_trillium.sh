#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
account="${SLURM_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$(cd "$script_dir/../.." && pwd)}"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
run_dir="${H2O_631G_PARENT_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/h2o/6-31g/parent_curve_$(date +%Y%m%d_%H%M%S)}"
array_spec="${H2O_631G_PARENT_ARRAY_SPEC:-0-12%13}"
equilibrium_dir="${H2O_631G_EQUILIBRIUM_REFERENCE_DIR:-}"

mkdir -p "$run_dir/logs"
export_values="ALL,LAS_PROJECT_DIR=$project_dir,LAS_VENV=$venv_dir,H2O_631G_PARENT_CURVE_RUN_DIR=$run_dir"
if [[ -n "$equilibrium_dir" ]]; then
    export_values+=",H2O_631G_EQUILIBRIUM_REFERENCE_DIR=$equilibrium_dir"
fi

array_job="$(sbatch --parsable \
    --account="$account" \
    --array="$array_spec" \
    --export="$export_values" \
    --output="$run_dir/logs/array_%A_%a.out" \
    "$script_dir/run_array_task_trillium.sh")"

aggregate_job="$(sbatch --parsable \
    --account="$account" \
    --dependency="afterok:$array_job" \
    --export="$export_values" \
    --output="$run_dir/logs/aggregate_%j.out" \
    "$script_dir/aggregate_trillium.sh")"

echo "H2O/6-31G parent-reference array: $array_job"
echo "Array specification: $array_spec"
echo "Aggregation job: $aggregate_job"
echo "Run directory: $run_dir"
echo "Monitor: squeue -j $array_job,$aggregate_job"
