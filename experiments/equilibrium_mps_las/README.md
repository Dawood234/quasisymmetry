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

After pulling the branch on the cluster, run directly from this directory:

```bash
cd ~/quasisymmetry/experiments/equilibrium_mps_las
```

Creating a separate bundle is optional. To create one:

Create the bundle locally:

```bash
./package_experiment.sh
```

Copy and unpack the printed archive on the cluster. The shared project clone
remains separate and is selected with `LAS_PROJECT_DIR`.

## H2O Launch

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

## N2 Launch

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
