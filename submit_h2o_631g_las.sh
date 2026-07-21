#!/usr/bin/env bash
#SBATCH --job-name=las_h2o_631g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/las_h2o_631g_%j.out
#SBATCH --export=ALL

set -euo pipefail

# Slurm copies the submitted script to a node-local spool directory.  Use the
# submission directory, rather than BASH_SOURCE, to find the checkout.
PROJECT_DIR="${LAS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
if [[ ! -f "${PROJECT_DIR}/run_h2o_631g_las.py" ]]; then
    echo "LAS project directory does not contain run_h2o_631g_las.py: ${PROJECT_DIR}" >&2
    exit 2
fi
# Set LAS_RUN_DIR when resubmitting after a walltime interruption.  Otherwise
# every new Slurm allocation gets its own provenance-preserving output folder.
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
RUN_DIR="${LAS_RUN_DIR:-${SCRATCH_ROOT}/alris/quasisymmetry/h2o/6-31g/job_${SLURM_JOB_ID}}"
mkdir -p "${RUN_DIR}"

if [[ -z "${LAS_VENV:-}" ]]; then
    echo "LAS_VENV must point to the group Python virtual environment." >&2
    exit 2
fi
if [[ ! -x "${LAS_VENV}/bin/python" ]]; then
    echo "Python not found at ${LAS_VENV}/bin/python" >&2
    exit 2
fi

# Use the cluster MPI stack that provides mpi4py for metrics.py.
module load openmpi mpi4py
source "${LAS_VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: ${PROJECT_DIR}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Run directory: ${RUN_DIR}"
echo "Restart command: LAS_PROJECT_DIR=${PROJECT_DIR} LAS_RUN_DIR=${RUN_DIR} sbatch ${PROJECT_DIR}/submit_h2o_631g_las.sh"
echo "Python: $(command -v python)"
python --version

python - <<'PY'
import ffsim
from mpi4py import MPI
import numpy
import openfermion
import pyscf
import scipy
from pyblock2.driver.core import DMRGDriver
print("Dependency imports passed")
PY

cd "${PROJECT_DIR}"
srun python -u run_h2o_631g_las.py \
    --run_dir "${RUN_DIR}" \
    --resume \
    --cpus "${SLURM_CPUS_PER_TASK}" \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
