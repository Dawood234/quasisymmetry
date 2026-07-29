#!/usr/bin/env bash
#SBATCH --account=rrg-izmaylov
#SBATCH --job-name=las_eq_mps_tri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
#SBATCH --output=las_eq_mps_trillium_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davood.dar@utoronto.ca
#SBATCH --signal=B:USR1@300

set -euo pipefail

walltime_warning_received=0
handle_walltime_warning() {
    walltime_warning_received=1
    echo
    echo "[walltime] Slurm sent USR1 at $(date --iso-8601=seconds)."
    echo "[walltime] The workflow will keep running until the scheduler stops it."
    echo "[walltime] Atomic candidate and generator checkpoints are safe to reuse with --resume."
}
trap handle_walltime_warning USR1

if [[ $# -eq 0 ]]; then
    echo "Pass the run_equilibrium_las.py arguments after the launcher." >&2
    exit 2
fi
if [[ -z "${SCRATCH:-}" || ! -d "$SCRATCH" ]]; then
    echo "Trillium SCRATCH is unavailable; submit from a Trillium login node." >&2
    exit 2
fi

module load scipy-stack/2026a

SLURM_ACCOUNT="${SLURM_JOB_ACCOUNT:-rrg-izmaylov}"
DEFAULT_PROJECT_DIR="$HOME/links/projects/$SLURM_ACCOUNT/$USER/quasisymmetry"
LAS_PROJECT_DIR="${LAS_PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
LAS_VENV="${LAS_VENV:-$HOME/las-env-trillium}"
EXPERIMENT_DIR="$LAS_PROJECT_DIR/experiments/equilibrium_mps_las"
TOTAL_THREADS="${SLURM_CPUS_PER_TASK:-192}"

if [[ ! -d "$LAS_PROJECT_DIR" ]]; then
    echo "LAS_PROJECT_DIR does not exist: $LAS_PROJECT_DIR" >&2
    exit 2
fi
if [[ ! -f "$EXPERIMENT_DIR/run_equilibrium_las.py" ]]; then
    echo "Experiment driver does not exist: $EXPERIMENT_DIR/run_equilibrium_las.py" >&2
    exit 2
fi
if [[ ! -x "$LAS_VENV/bin/python" ]]; then
    echo "LAS_VENV Python does not exist: $LAS_VENV/bin/python" >&2
    exit 2
fi

arguments=("$@")
has_threads=false
for argument in "${arguments[@]}"; do
    if [[ "$argument" == "--threads" || "$argument" == --threads=* ]]; then
        has_threads=true
        break
    fi
done
if [[ "$has_threads" == false ]]; then
    arguments+=(--threads "$TOTAL_THREADS")
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$TOTAL_THREADS"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LAS_PROJECT_DIR

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Cluster: Trillium"
echo "Account: $SLURM_ACCOUNT"
echo "Project: $LAS_PROJECT_DIR"
echo "Experiment: $EXPERIMENT_DIR"
echo "Submission directory: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "Scratch: $SCRATCH"
echo "Python: $LAS_VENV/bin/python"
echo "CPUs: $TOTAL_THREADS"
echo "Arguments: ${arguments[*]}"
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

srun --ntasks=1 --cpus-per-task="$TOTAL_THREADS" \
    "$LAS_VENV/bin/python" -u "$EXPERIMENT_DIR/run_equilibrium_las.py" \
    --project_dir "$LAS_PROJECT_DIR" "${arguments[@]}" &
workflow_pid=$!

# A trapped signal interrupts wait but not the running srun step. Wait again
# until the workflow exits or Slurm enforces the actual walltime limit.
set +e
while true; do
    wait "$workflow_pid"
    workflow_status=$?
    if kill -0 "$workflow_pid" 2>/dev/null; then
        continue
    fi
    break
done
set -e

if [[ "$workflow_status" -ne 0 ]]; then
    if [[ "$walltime_warning_received" -eq 1 ]]; then
        echo "[walltime] Workflow stopped after the warning; resubmit the same command with --resume." >&2
    fi
    exit "$workflow_status"
fi

echo
echo "Finished: $(date --iso-8601=seconds)"
