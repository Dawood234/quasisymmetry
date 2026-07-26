#!/usr/bin/env bash
#SBATCH --account=def-izmaylov
#SBATCH --job-name=las_eq_mps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/las_eq_mps_%j.out
#SBATCH --signal=B:USR1@300

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Pass the run_equilibrium_las.py arguments after the launcher." >&2
    exit 2
fi

LAS_PROJECT_DIR="${LAS_PROJECT_DIR:-$HOME/quasisymmetry}"
LAS_VENV="${LAS_VENV:-$HOME/las-env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$LAS_PROJECT_DIR" ]]; then
    echo "LAS_PROJECT_DIR does not exist: $LAS_PROJECT_DIR" >&2
    exit 2
fi
if [[ ! -x "$LAS_VENV/bin/python" ]]; then
    echo "LAS_VENV Python does not exist: $LAS_VENV/bin/python" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS=1
export LAS_PROJECT_DIR

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: $LAS_PROJECT_DIR"
echo "Experiment: $SCRIPT_DIR"
echo "Python: $LAS_VENV/bin/python"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-32}"
echo "Arguments: $*"
echo

"$LAS_VENV/bin/python" - <<'PY'
import importlib
import sys

required = ("numpy", "scipy", "pyscf", "pyblock2", "openfermion", "ffsim")
for name in required:
    importlib.import_module(name)
print("Dependency imports passed")
print("Python", sys.version.replace("\n", " "))
PY

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
    "$LAS_VENV/bin/python" -u "$SCRIPT_DIR/run_equilibrium_las.py" \
    --project_dir "$LAS_PROJECT_DIR" "$@"

echo
echo "Finished: $(date --iso-8601=seconds)"
