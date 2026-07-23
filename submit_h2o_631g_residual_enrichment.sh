#!/usr/bin/env bash
#SBATCH --job-name=las_h2o_residual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/las_h2o_residual_%j.out
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${LAS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
if [[ ! -f "${PROJECT_DIR}/selected_clifford_residual_enrichment.py" ]]; then
    echo "LAS project directory is missing selected_clifford_residual_enrichment.py" >&2
    exit 2
fi
if [[ -z "${LAS_RUN_DIR:-}" ]]; then
    echo "LAS_RUN_DIR must identify the completed H2O/6-31G run." >&2
    exit 2
fi
if [[ -z "${LAS_VENV:-}" || ! -x "${LAS_VENV}/bin/python" ]]; then
    echo "LAS_VENV must identify the LAS Python environment." >&2
    exit 2
fi

SOURCE_METRICS="${LAS_SOURCE_METRICS:-${LAS_RUN_DIR}/final_krylov_metrics_s32.json}"
WORK_DIR="${LAS_RESIDUAL_WORK_DIR:-${LAS_RUN_DIR}/final_residual_enrichment_s32}"
OUTPUT="${LAS_RESIDUAL_OUTPUT:-${LAS_RUN_DIR}/final_residual_metrics_s32.json}"
if [[ ! -f "${SOURCE_METRICS}" ]]; then
    echo "Source Krylov metrics do not exist: ${SOURCE_METRICS}" >&2
    exit 2
fi

source "${LAS_VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

printf 'Host: %s\n' "$(hostname)"
printf 'Started: %s\n' "$(date --iso-8601=seconds)"
printf 'Project: %s\n' "${PROJECT_DIR}"
printf 'Existing run directory: %s\n' "${LAS_RUN_DIR}"
printf 'Source metrics: %s\n' "${SOURCE_METRICS}"
printf 'Enrichment work directory: %s\n' "${WORK_DIR}"
printf 'Output: %s\n' "${OUTPUT}"
printf 'Python: %s\n' "$(command -v python)"
python --version

cd "${PROJECT_DIR}"
srun python -u selected_clifford_residual_enrichment.py \
    --run_dir "${LAS_RUN_DIR}" \
    --source_metrics "${SOURCE_METRICS}" \
    --work_dir "${WORK_DIR}" \
    --outname "${OUTPUT}" \
    --max_macrocycles "${LAS_RESIDUAL_CYCLES:-10}" \
    --existing_sectors_per_cycle "${LAS_RESIDUAL_EXISTING_SECTORS:-16}" \
    --new_sectors_per_cycle "${LAS_RESIDUAL_NEW_SECTORS:-16}" \
    --chain_depth "${LAS_RESIDUAL_CHAIN_DEPTH:-4}" \
    --max_dimension "${LAS_RESIDUAL_MAX_K:-2000}" \
    --energy_tolerance_mha "${LAS_RESIDUAL_ENERGY_TOLERANCE_MHA:-0.0}" \
    --resume \
    "$@"

printf 'Finished: %s\n' "$(date --iso-8601=seconds)"
