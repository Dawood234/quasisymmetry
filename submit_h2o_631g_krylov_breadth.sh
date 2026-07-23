#!/usr/bin/env bash
#SBATCH --job-name=las_h2o_kbreadth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/las_h2o_krylov_breadth_%j.out
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${LAS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
if [[ ! -f "${PROJECT_DIR}/run_krylov_sector_breadth.py" ]]; then
    echo "LAS project directory is missing run_krylov_sector_breadth.py" >&2
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

module load openmpi mpi4py
source "${LAS_VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: ${PROJECT_DIR}"
echo "Existing run directory: ${LAS_RUN_DIR}"
echo "Python: $(command -v python)"
python --version

cd "${PROJECT_DIR}"
srun python -u run_krylov_sector_breadth.py \
    --run_dir "${LAS_RUN_DIR}" \
    --sector_counts "${KRYLOV_SECTOR_COUNTS:-24,32}" \
    --krylov_depths "${KRYLOV_DEPTHS:-1,2,4,8,12,16,24}" \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
