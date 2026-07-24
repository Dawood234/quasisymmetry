#!/usr/bin/env bash
#SBATCH --job-name=n2_631g_parent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/%u/las_n2_631g_parent_%A_%a.out
#SBATCH --export=ALL

set -euo pipefail

# Run independent M=350 and M=500 parent references. Keeping them separate
# lets the LAS selection/optimization workflow proceed at the same time.
BOND_DIMS=(350 500)
BOND_DIM="${BOND_DIMS[${SLURM_ARRAY_TASK_ID}]}"
N_SWEEPS="${N2_PARENT_SWEEPS:-20}"
N2_BOND="${N2_BOND:-1.0977}"
N2_REORDER="${N2_REORDER:-fiedler}"
N2_ENERGY_TOL="${N2_ENERGY_TOL:-1e-8}"
N2_DAVIDSON_THRESHOLD="${N2_DAVIDSON_THRESHOLD:-1e-10}"
N2_TWO_TO_ONE="${N2_TWO_TO_ONE:-12}"
N2_DMRG_IPRINT="${N2_DMRG_IPRINT:-1}"
N2_PREP_BOND="${N2_PREP_BOND:-0}"
N2_PREP_SWEEPS="${N2_PREP_SWEEPS:-8}"

PROJECT_DIR="${LAS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
if [[ ! -f "${PROJECT_DIR}/solve_dmrg.py" ]]; then
    echo "LAS project directory does not contain solve_dmrg.py: ${PROJECT_DIR}" >&2
    exit 2
fi

if [[ -z "${LAS_VENV:-}" ]]; then
    echo "LAS_VENV must point to the group Python virtual environment." >&2
    exit 2
fi
if [[ ! -x "${LAS_VENV}/bin/python" ]]; then
    echo "Python not found at ${LAS_VENV}/bin/python" >&2
    exit 2
fi

SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
REFERENCE_ROOT="${N2_PARENT_ROOT:-${SCRATCH_ROOT}/alris/quasisymmetry/n2/6-31g/r_${N2_BOND}/parent_reference}"
TASK_DIR="${REFERENCE_ROOT}/M${BOND_DIM}"
CHECKPOINT="${TASK_DIR}/n2_${N2_BOND}_6-31g_d2h.chk"
MPS_DIR="${TASK_DIR}/mps"
RESULT="${TASK_DIR}/parent_M${BOND_DIM}.txt"
mkdir -p "${TASK_DIR}"

source "${LAS_VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: ${PROJECT_DIR}"
echo "Reference root: ${REFERENCE_ROOT}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Bond dimension: ${BOND_DIM}"
echo "Sweeps: ${N_SWEEPS}"
echo "N-N distance: ${N2_BOND} Angstrom"
echo "Orbital reorder: ${N2_REORDER}"
echo "Energy tolerance: ${N2_ENERGY_TOL}"
echo "Davidson threshold: ${N2_DAVIDSON_THRESHOLD}"
echo "2-site sweeps before 1-site: ${N2_TWO_TO_ONE}"
echo "Optional warm-up bond dimension: ${N2_PREP_BOND}"
echo "Python: $(command -v python)"
python --version

python - <<'PY'
from pyblock2.driver.core import DMRGDriver
import numpy
import pyscf
print("Dependency imports passed")
PY

cd "${PROJECT_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo
    echo "=== Build N2/6-31G D2h checkpoint for M=${BOND_DIM} ==="
    srun python -u make_pyscf_hamiltonian.py \
        n2 "${N2_BOND}" \
        --basis 6-31g \
        --point_group D2h \
        --output "${CHECKPOINT}"
else
    echo "Reusing checkpoint: ${CHECKPOINT}"
fi

if [[ -f "${RESULT}" ]] && grep -q '^E_DMRG ' "${RESULT}"; then
    echo "Reference already complete; reusing ${RESULT}"
    grep '^E_DMRG ' "${RESULT}"
else
    initial_tag=()
    if (( N2_PREP_BOND > 0 && N2_PREP_BOND < BOND_DIM )); then
        prep_result="${TASK_DIR}/parent_PREP_M${N2_PREP_BOND}.txt"
        echo
        echo "=== Warm-up DMRG M=${N2_PREP_BOND} ==="
        prep_command=(
            srun python -u solve_dmrg.py
            "${CHECKPOINT}" \
            --bond_dim "${N2_PREP_BOND}"
            --n_sweeps "${N2_PREP_SWEEPS}"
            --energy_tol 1e-6
            --davidson_threshold 1e-8
            --dmrg_iprint "${N2_DMRG_IPRINT}"
            --mps_tag PREP
            --n_threads "${SLURM_CPUS_PER_TASK}"
            --store_dir "${MPS_DIR}"
            --outname "${prep_result}"
        )
        if [[ "${N2_REORDER}" != "none" ]]; then
            prep_command+=(--reorder "${N2_REORDER}")
        fi
        "${prep_command[@]}"
        initial_tag=(--initial_mps_tag PREP)
    fi

    echo
    echo "=== Parent DMRG M=${BOND_DIM} ==="
    command=(
        srun python -u solve_dmrg.py
        "${CHECKPOINT}"
        --bond_dim "${BOND_DIM}"
        --n_sweeps "${N_SWEEPS}"
        --energy_tol "${N2_ENERGY_TOL}"
        --davidson_threshold "${N2_DAVIDSON_THRESHOLD}"
        --twosite_to_onesite "${N2_TWO_TO_ONE}"
        --dmrg_iprint "${N2_DMRG_IPRINT}"
        --mps_tag "M${BOND_DIM}"
        --n_threads "${SLURM_CPUS_PER_TASK}"
        --store_dir "${MPS_DIR}"
        --outname "${RESULT}"
    )
    command+=("${initial_tag[@]}")
    if [[ "${N2_REORDER}" != "none" ]]; then
        command+=(--reorder "${N2_REORDER}")
    fi
    "${command[@]}"
fi

echo
echo "Result:"
grep '^E_DMRG ' "${RESULT}"
echo "Finished: $(date --iso-8601=seconds)"
