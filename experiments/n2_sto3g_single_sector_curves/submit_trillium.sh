#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
slurm_account="${SLURM_ACCOUNT:-rrg-izmaylov}"
project_dir="${LAS_PROJECT_DIR:-$(cd "$script_dir/../.." && pwd)}"
venv_dir="${LAS_VENV:-$HOME/las-env-trillium}"
run_dir="${N2_CURVE_RUN_DIR:-$SCRATCH/alris/quasisymmetry/n2/sto-3g/single_sector_initial_lowest_curves_20260801}"
array_spec="${N2_ARRAY_SPEC:-0-20%21}"

mkdir -p "$run_dir/logs"
export_values="ALL,LAS_PROJECT_DIR=$project_dir,LAS_VENV=$venv_dir,N2_CURVE_RUN_DIR=$run_dir"

array_job="$(sbatch --parsable \
    --account="$slurm_account" \
    --array="$array_spec" \
    --export="$export_values" \
    --output="$run_dir/logs/array_%A_%a.out" \
    "$script_dir/run_array_task_trillium.sh")"

aggregate_job="$(sbatch --parsable \
    --account="$slurm_account" \
    --dependency="afterok:$array_job" \
    --export="$export_values" \
    --output="$run_dir/logs/aggregate_%j.out" \
    "$script_dir/run_aggregate_trillium.sh")"

echo "N2 array job: $array_job"
echo "Array specification: $array_spec"
echo "Aggregation job: $aggregate_job (starts after every array task succeeds)"
echo "Run directory: $run_dir"
echo "Monitor with: squeue -j $array_job,$aggregate_job"
