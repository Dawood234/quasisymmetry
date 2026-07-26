# Equilibrium H2O/N2 6-31G MPS LAS Experiment

This is a temporary tracked cluster-staging copy of the isolated experiment.
It imports the repository's production modules but does not modify them.
Remove this whole directory after H2O validation and the N2 production run.

## Scientific Path

1. Validate the checkpoint, proxy MPS, certified reference, orbital ordering,
   point group, electron count, and fingerprints.
2. Score all seniority-plus-quartet candidates with MPS-native NC.
3. Select seven GF(2)-independent generators.
4. Screen 8-16 sector labels by recursive MPS projection while recording the
   recovered dominant-determinant norm as a diagnostic.
5. Optimize switching-sector decoupled energy with analytic RDM gradients.
6. Reselect generators and repeat until row space and energy stabilize, with
   at most three macrocycles.
7. Converge the final anchor sector at M=100, 200, 350, and 500; add M=750
   when M=350 and M=500 differ by more than 0.2 mHa.
8. Project `(H-E_dec)|phi_0>` into external sectors and form coupling-seeded
   sector-local MPS Krylov chains.
9. Solve `H_K c = E S_K c`, fit the Ritz MPS, form its residual, and enrich
   the strongest missing or under-resolved sectors.
10. After the coupled space is fixed, export only represented diagonal and
    inter-sector tapered Pauli LCUs. No sparse matrix is built.

H2O runs the established determinant-space comparator unless
`--determinant_result` reuses a prior comparator result or
`--skip_determinant_evaluator` is explicitly supplied. N2 requires a passing
H2O `summary.json` unless it is explicitly launched as an unvalidated smoke
test.

## Cluster Bundle

The downloaded parent-reference artifacts are stored locally under
`parent_references/`:

```text
parent_references/h2o/job_50658575/
parent_references/n2/job_50805667/
```

They include the molecular checkpoints, certified reference summaries, and
reusable Block2 MPS tags. The directory is excluded through `.git/info/exclude`
because generated wavefunctions must not be committed to the software
repository.

Code and parent data use separate transfer mechanisms:

```bash
git pull
./package_parent_references.sh
```

`git pull` supplies the tracked experiment code.
`package_parent_references.sh` creates an artifact-only archive containing
`parent_references/` and its `SHA256SUMS`. Transfer that archive to Trillium
with Globus or `scp` and unpack it under `$SCRATCH`. This avoids both code
duplication and rerunning either parent DMRG calculation.

Run the experiment from the repository directory:

```bash
cd ~/quasisymmetry/experiments/equilibrium_mps_las
```

Point `H2O_PARENT` and `N2_PARENT` to the separately extracted artifact tree.

## Cluster Launchers

- `submit_equilibrium_las.sh` is the existing Fir launcher. It remains the
  32-CPU, 64-GiB path used by the current Fir jobs.
- `submit_equilibrium_las_trillium.sh` is the Trillium launcher. It requests
  one complete 192-core CPU node under `rrg-izmaylov`; Trillium supplies the
  node's full memory, so the script does not request a smaller memory slice.

Trillium has separate storage and software from Fir. Transfer the checkpoint,
proxy MPS, and reference-result artifacts to Trillium before launching, and
create a Trillium virtual environment rather than copying the Fir environment.
The Trillium launcher loads `scipy-stack/2026a` and defaults to
`$HOME/las-env-trillium`. Trillium submits with `--export=NONE` and executes a
spooled copy of the batch script. The launcher therefore derives the persistent
repository as
`$HOME/links/projects/$SLURM_JOB_ACCOUNT/$USER/quasisymmetry` rather than using
an exported variable or the batch script's runtime location.

Trillium home directories are read-only on compute nodes. Submit from a
directory under `$SCRATCH` so the relative Slurm output file can be created:

```bash
cd "$SCRATCH"
export LAS_PROJECT_DIR="/home/$USER/links/projects/rrg-izmaylov/$USER/quasisymmetry"
export LAS_VENV="$HOME/las-env-trillium"
```

The Trillium launcher supplies `--threads 192` unless it is explicitly given.
With `--candidate_workers 10`, candidate scoring uses about 19 threads per
worker. With `--sector_workers 4`, projector and Krylov work uses 48 threads
per worker.

## H2O Launch On Fir

```bash
export LAS_VENV="$HOME/las-env"
export LAS_PROJECT_DIR="$HOME/quasisymmetry"

sbatch submit_equilibrium_las.sh \
  --system h2o \
  --checkpoint /scratch/$USER/artifacts/h2o_631g_c2v.chk \
  --proxy_mps /scratch/$USER/artifacts/h2o_proxy_mps \
  --reference_result /scratch/$USER/artifacts/h2o_reference.json \
  --run_dir /scratch/$USER/alris/equilibrium_mps_las/h2o \
  --sector_workers 1 \
  --resume
```

Use `--sector_workers 1` for the first H2O validation. After that serial path
passes, resume or launch a fresh matched run with `--sector_workers 4`; each
worker receives eight threads on the 32-CPU launcher.

## H2O Launch On Trillium

```bash
PARENT_ROOT="$SCRATCH/alris/equilibrium_mps_parent_references"
H2O_PARENT="$PARENT_ROOT/parent_references/h2o/job_50658575"
H2O_RUN="$SCRATCH/alris/quasisymmetry/equilibrium_mps_las/h2o_631g"
TRILLIUM_LAUNCHER="$LAS_PROJECT_DIR/experiments/equilibrium_mps_las/submit_equilibrium_las_trillium.sh"

sbatch "$TRILLIUM_LAUNCHER" \
  --system h2o \
  --checkpoint "$H2O_PARENT/h2o_0.958_104.5_6-31g_c2v.chk" \
  --proxy_mps "$H2O_PARENT/parent_mps" \
  --proxy_tag M200 \
  --reference_result "$H2O_PARENT/parent_reference_summary.json" \
  --run_dir "$H2O_RUN" \
  --candidate_workers 10 \
  --sector_workers 4 \
  --resume
```

## N2 Launch On Fir

```bash
sbatch submit_equilibrium_las.sh \
  --system n2 \
  --checkpoint /scratch/$USER/artifacts/n2_631g_d2h.chk \
  --proxy_mps /scratch/$USER/artifacts/n2_proxy_mps \
  --reference_result /scratch/$USER/artifacts/n2_reference.json \
  --h2o_validation_summary /scratch/$USER/alris/equilibrium_mps_las/h2o/summary.json \
  --run_dir /scratch/$USER/alris/equilibrium_mps_las/n2 \
  --sector_workers 4 \
  --resume
```

## N2 Launch On Trillium

Run N2 only after H2O writes a passing `summary.json`:

```bash
N2_PARENT="$PARENT_ROOT/parent_references/n2/job_50805667"
N2_RUN="$SCRATCH/alris/quasisymmetry/equilibrium_mps_las/n2_631g"

sbatch "$TRILLIUM_LAUNCHER" \
  --system n2 \
  --checkpoint "$N2_PARENT/n2_1.0977_6-31g_d2h.chk" \
  --proxy_mps "$N2_PARENT/parent_mps" \
  --proxy_tag M200 \
  --reference_result "$N2_PARENT/parent_reference_summary.json" \
  --h2o_validation_summary "$H2O_RUN/summary.json" \
  --run_dir "$N2_RUN" \
  --candidate_workers 10 \
  --sector_workers 4 \
  --resume
```

The parent checkpoint, proxy MPS, and certified reference are read-only
inputs. All mutable Block2 stores and operation records are created below
`--run_dir`. Production runs require the selected proxy tag to have bond
dimension 100 or 200. Isolated NC workers cap their copied Block2 stack at
4 GiB so eight N2 workers fit safely inside the 64 GiB job allocation.

## Restart

Repeat the same command with `--resume`. Artifact hashes must match the first
launch. Completed macrocycles, DMRG stages, fitted MPS operations, coupled
matrix rows, and enrichment cycles are reused.

## Validation

Run pure local tests without Block2:

```bash
python -m pytest -q tests/test_pure_math.py tests/test_toy_workflow.py
```

Run the Block2 integration checks in the cluster environment:

```bash
python -m pytest -q tests/test_block2_contracts.py
```

Run the MPS-versus-exact STO-3G validator separately for H2O and N2:

```bash
python validate_sto3g.py \
  --project_dir "$LAS_PROJECT_DIR" \
  --proxy_mps /path/to/sto3g_parent_mps \
  --proxy_tag GS \
  --run_dir /path/to/validation_run
```

The local validation completed for both molecules. Recursive projector
weights agreed with exact determinant weights within `1.8e-15`; the
two-vector MPS Krylov Ritz energies agreed with exact determinant Krylov
energies within `6.9e-13` Ha for H2O and `1.6e-12` Ha for N2.
