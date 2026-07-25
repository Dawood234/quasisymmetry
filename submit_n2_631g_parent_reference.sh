#!/usr/bin/env bash
#SBATCH --job-name=n2_631g_parent
#SBATCH --account=def-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/las_n2_631g_parent_%j.out
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${LAS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
if [[ ! -f "${PROJECT_DIR}/run_n2_631g_parent_reference.py" ]]; then
    echo "LAS project directory is invalid: ${PROJECT_DIR}" >&2
    exit 2
fi

if [[ -z "${LAS_VENV:-}" ]]; then
    echo "LAS_VENV must point to the cluster Python virtual environment." >&2
    exit 2
fi
if [[ ! -x "${LAS_VENV}/bin/python" ]]; then
    echo "Python not found at ${LAS_VENV}/bin/python" >&2
    exit 2
fi

SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
RUN_DIR="${LAS_RUN_DIR:-${SCRATCH_ROOT}/alris/quasisymmetry/n2/6-31g/parent_reference/job_${SLURM_JOB_ID}}"
PERSISTENT_STORE="${RUN_DIR}/parent_mps"
LOCAL_ROOT="${SLURM_TMPDIR:-${RUN_DIR}/local_work}"
WORKING_STORE="${LOCAL_ROOT}/parent_mps"
mkdir -p "${RUN_DIR}" "${PERSISTENT_STORE}" "${WORKING_STORE}"

THREADS="${N2_PARENT_THREADS:-${SLURM_CPUS_PER_TASK}}"
MODE="${N2_PARENT_MODE:-production}"
BOND_DIMS="${N2_PARENT_BOND_DIMS:-100,200,350,500}"
STAGE_SWEEPS="${N2_PARENT_STAGE_SWEEPS:-4,4,6,8}"
REVERSE_DIMS="${N2_PARENT_REVERSE_DIMS:-500,500,350,350,250,250,200,200}"
SYMMETRY_MODE="${N2_PARENT_SYMMETRY:-su2}"
ORDERING="${N2_PARENT_ORDERING:-fiedler}"
STACK_MEM_GB="${N2_PARENT_STACK_MEM_GB:-8}"
N2_BOND="${N2_BOND:-1.0977}"

source "${LAS_VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${LOCAL_ROOT}/matplotlib"
export XDG_CACHE_HOME="${LOCAL_ROOT}/cache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

persist_store() {
    if [[ -d "${WORKING_STORE}" ]]; then
        echo "Persisting node-local MPS files to ${PERSISTENT_STORE}"
        if command -v rsync >/dev/null 2>&1; then
            rsync -a "${WORKING_STORE}/" "${PERSISTENT_STORE}/" || true
        else
            cp -a "${WORKING_STORE}/." "${PERSISTENT_STORE}/" || true
        fi
    fi
}
trap persist_store EXIT TERM INT

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: ${PROJECT_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Node-local MPS store: ${WORKING_STORE}"
echo "Persistent MPS store: ${PERSISTENT_STORE}"
echo "N-N distance: ${N2_BOND} Angstrom"
echo "Mode: ${MODE}"
echo "Block2 spin symmetry: ${SYMMETRY_MODE}"
echo "Orbital ordering: ${ORDERING}"
echo "Threads: ${THREADS} Block2, 1 MKL"
echo "Bond dimensions: ${BOND_DIMS}"
echo "Stage sweeps: ${STAGE_SWEEPS}"
echo "Reverse schedule: ${REVERSE_DIMS}"
echo "Python: $(command -v python)"
python --version

python - <<'PY'
from pyblock2.driver.core import DMRGDriver
import numpy
import pyscf
import scipy
print("Dependency imports passed")
PY

cd "${PROJECT_DIR}"

command=(
    srun python -u run_n2_631g_parent_reference.py
    --run_dir "${RUN_DIR}"
    --working_store "${WORKING_STORE}"
    --persistent_store "${PERSISTENT_STORE}"
    --mode "${MODE}"
    --bond_length "${N2_BOND}"
    --symmetry_mode "${SYMMETRY_MODE}"
    --ordering "${ORDERING}"
    --n_threads "${THREADS}"
    --n_mkl_threads 1
    --stack_mem_gb "${STACK_MEM_GB}"
    --bond_dims "${BOND_DIMS}"
    --stage_sweeps "${STAGE_SWEEPS}"
    --reverse_bond_dims "${REVERSE_DIMS}"
)
command+=("$@")

echo "Restart command:"
echo "LAS_PROJECT_DIR=${PROJECT_DIR} LAS_VENV=${LAS_VENV} LAS_RUN_DIR=${RUN_DIR} sbatch ${PROJECT_DIR}/submit_n2_631g_parent_reference.sh --resume"
echo
"${command[@]}"

echo "Finished: $(date --iso-8601=seconds)"
