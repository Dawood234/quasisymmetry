import argparse
import bisect
import json
import time
from math import comb
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse.linalg
from tqdm import tqdm
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from chemistry import CHEMICAL_PRECISION, fcidump_data, load_moldata
from optimize_symmetries import (
    commutator_cost,
    get_fci,
    parity_matrix_to_quasisymmetries,
    x_to_rotation,
)
from src.energy_diagnostics import (
    coupled_energy_perturbation,
    reference_coupled_energy_k,
    sector_data_from_gs_pairs,
    state_labels_for_columns,
)
from src.davidson_solver import solve_sector_davidson
from src.sector_utils import subspace_matrix, symmetry_sectors
from src.workflow_cli import (
    add_metrics_workflow_args,
    print_workflow_banner,
)
from src.clifford_sectors import (
    build_clifford_frame,
    candidate_hamiltonian,
    candidate_reference_weights,
    ci_vector_to_jw_state,
    coupled_energy_curve,
    load_symmetry_manifest,
    molecular_hamiltonian_to_jw,
    pauli_lcu_is_hermitian,
    parse_sector_labels,
    perturbative_coupled_energy_curve,
    physical_clifford_basis,
    physical_clifford_matrix,
    qubit_operator_to_data,
    reference_candidate_order,
    restricted_operator_matrix,
    sector_state_candidates,
    solve_physical_clifford_sector,
    solve_tapered_sector,
    tapered_operator,
    z_symmetries_from_parity_matrix,
)

# Used by MPI worker processes (must be importable at module level).
import ffsim


def submatrix_eigenvalues_to_target(A: np.ndarray, e_target: float):
    """Start in the upper left corner of A, take a KxK block and calculate its
    lowest eignvalue. Return the smallest K that yields energy below e_target
    or -1 if no such thing can be found, and the vector that does it"""
    e_full, v_full = scipy.sparse.linalg.eigsh(A, which="SA", k=1)
    energies = np.zeros(A.shape[0])
    energies[0] = A[0, 0].real

    if e_full > e_target:
        return -1, v_full
    elif A[0, 0] < e_target:
        v = np.zeros(A.shape[0])
        v[0] = 1
        return 1, v
    else:
        order = np.argsort(abs(v_full.flatten()))[::-1]
        B = A[np.ix_(order, order)]
        for vec_count in tqdm(range(2, B.shape[0] + 1)):
            submatrix = B[:vec_count, :vec_count]
            e, v = np.linalg.eigh(submatrix)
            energies[vec_count - 1] = e[0]
            if e[0] < e_target:
                y = np.zeros(B.shape[0], dtype="complex")
                y[:vec_count] = v[:, 0]
                return vec_count, y

        else:
            plt.plot(energies - e_target)
            plt.yscale("log")
            plt.axhline(e_full - e_target)
            plt.show()
            raise ValueError("this should never happen")


def selected_column_solver(A: np.ndarray, e_target, thr=1e-8, start="zero"):
    if start == "zero":
        starting_index = 0
    elif start == "energy":
        starting_index = np.argmin(np.diag(A))
    else:
        raise ValueError()
    vector_count = -1
    current_vector = np.zeros(A.shape[0])
    current_vector[starting_index] = 1
    current_round = 0
    current_dimension = 1
    if current_vector.T.conj() @ A @ current_vector < e_target:
        return 1, current_vector
    while vector_count == -1:
        current_round += 1
        if current_round > 1000:
            raise ValueError("MaxIter")
        print("SCI-like round ", current_round)
        current_indices = np.where(abs(A @ current_vector) + abs(current_vector) > thr)
        print("dimension ", len(current_indices[0]))
        if len(current_indices[0]) == current_dimension:
            print("stopping as nothing new found within thr")
            break
        current_dimension = len(current_indices[0])
        submatrix = A[np.ix_(current_indices[0], current_indices[0])]
        vector_count, v = submatrix_eigenvalues_to_target(submatrix, e_target)
        current_vector = np.zeros(A.shape[0], dtype="complex")
        current_vector[current_indices] = v.flatten()
        print("SCI-like energy", current_vector.T.conj() @ A @ current_vector)
    return vector_count, current_vector


def orthogonalize_degenerate(w, V, tol=1e-10):
    """scipy.sparse.linalg.eigsh sometimes returns non-orthogonal eigenvectors if they have
    degenerate eigenvalues. This function rectifies that."""
    V_orth = V.copy()

    start = 0
    while start < len(w):
        end = start + 1
        while end < len(w) and abs(w[end] - w[start]) < tol:
            end += 1

        # Orthogonalize this degenerate block
        Q, _ = scipy.linalg.qr(V[:, start:end], mode="economic")
        V_orth[:, start:end] = Q

        start = end
    return V_orth


def find_first_negative(f, N):
    domain = range(1, N + 1)
    index = bisect.bisect_left(domain, x=True, key=lambda x: f(x) < 0)
    if index < len(domain):
        return domain[index]
    return -1


def solve_eigs(data):
    # mpi4py can't pickle the rotated_h_linop, so reconstruct it on each worker.
    from mpi4py import MPI

    moldata = data["moldata"]
    rotated_h = data["rotated_h"]
    sector_bitstrings = data["sector_bitstrings"]
    rotated_h_linop = ffsim.linear_operator(
        rotated_h, norb=moldata.norb, nelec=moldata.nelec
    )

    h_subspace = subspace_matrix(rotated_h_linop, sector_bitstrings)
    if data["states_per_sector"] <= h_subspace.shape[0] - 2:
        w, v = scipy.sparse.linalg.eigsh(
            h_subspace, which="SA", k=data["states_per_sector"]
        )
        v = v[:, np.argsort(w)]
        w = np.sort(w)
        v_orth = orthogonalize_degenerate(w, v)
        sector_eigs = w, v_orth
    else:
        sector_eigs = np.linalg.eigh(h_subspace)

    return {
        "sector_label": data["sector_label"],
        "sector_eigs": sector_eigs,
        "rank": MPI.COMM_WORLD.Get_rank(),
        "hostname": MPI.Get_processor_name(),
    }


def solve_eigs_davidson(data):
    """MPI worker: PySCF Davidson on the same sector block as ``solve_eigs``."""
    from mpi4py import MPI

    moldata = data["moldata"]
    rotated_h = data["rotated_h"]
    sector_bitstrings = data["sector_bitstrings"]
    rotated_h_linop = ffsim.linear_operator(
        rotated_h, norb=moldata.norb, nelec=moldata.nelec
    )

    h_subspace = subspace_matrix(rotated_h_linop, sector_bitstrings)
    nroots = min(int(data["states_per_sector"]), h_subspace.shape[0])
    w, v, meta = solve_sector_davidson(
        h_subspace,
        nroots=nroots,
        tol=data.get("davidson_tol", 1e-12),
        max_cycle=data.get("davidson_max_cycle", 50),
        max_space=data.get("davidson_max_space", 12),
    )
    return {
        "sector_label": data["sector_label"],
        "sector_eigs": (w, v),
        "meta": meta,
        "rank": MPI.COMM_WORLD.Get_rank(),
        "hostname": MPI.Get_processor_name(),
    }


def _run_dmrg_from_oo_json(input_data, args, outname, out_data):
    """MPS-native metrics path using OO JSON (molpath / parity / rotation)."""
    from src.dmrg_diagnostics import format_metrics_report, run_dmrg_metrics
    from src.dmrg_solver import (
        Block2DMRGSolver,
        DMRGConfig,
        rotate_integrals,
    )

    molpath = Path(input_data["molpath"])
    parity_path = input_data.get("parity")
    if parity_path is None:
        raise SystemExit("DMRG metrics require a parity matrix in the OO JSON")
    parity_matrix = np.atleast_2d(np.loadtxt(parity_path, dtype=int))

    store_dir = args.wavefunction_dir
    if store_dir is None:
        store_dir = str(Path("wavefunctions") / (molpath.stem + "_metrics"))

    if molpath.suffix == ".chk":
        dumpdata = fcidump_data(str(molpath))
        base = Block2DMRGSolver.from_dumpdata(
            dumpdata, store_dir=None, n_threads=args.n_threads, save_integrals=False
        )
    else:
        base = Block2DMRGSolver.from_fcidump(
            molpath, store_dir=None, n_threads=args.n_threads, save_integrals=False
        )
    h1e, g2e, ecore = base.h1e, base.g2e, base.ecore
    n_elec, spin = base.n_elec, base.spin

    rotation = np.asarray(input_data.get("rotation", []), dtype=float)
    if rotation.size:
        from src.orbital_rotation import pairs_from_oo_data, params_to_U

        pairs = pairs_from_oo_data(input_data, h1e.shape[0])
        h1e, g2e = rotate_integrals(
            h1e, g2e, params_to_U(rotation, h1e.shape[0], pairs)
        )

    solver = Block2DMRGSolver(
        h1e=h1e,
        g2e=g2e,
        ecore=ecore,
        n_elec=n_elec,
        spin=spin,
        store_dir=store_dir,
        n_threads=args.n_threads,
        reorder=args.reorder,
    )
    states_per_sector = (
        args.states_per_sector if args.states_per_sector < 50 else 5
    )
    solve_start = time.perf_counter()
    report = run_dmrg_metrics(
        solver,
        parity_matrix,
        config=DMRGConfig(
            max_bond_dim=args.bond_dim,
            n_sweeps=args.dmrg_sweeps,
        ),
        penalty=args.penalty,
        max_sectors=args.max_sectors,
        states_per_sector=states_per_sector,
        compute_k=True,
        compute_entanglement=args.entanglement,
    )
    solve_time_s = time.perf_counter() - solve_start
    lines = format_metrics_report(report)
    for line in lines:
        print(line)

    out_data["backend"] = "dmrg"
    out_data["solver"] = "dmrg"  # alias for older consumers
    out_data["solve_time_s"] = float(solve_time_s)
    out_data["report_lines"] = lines
    out_data["E_FCI"] = report.e_reference
    out_data["E_decoupled"] = report.decoupled.e_decoupled
    out_data["dE"] = report.decoupled.dE
    if report.coupled is not None:
        out_data["E_coupled"] = report.coupled.e_coupled
        out_data["K"] = report.coupled.k
        out_data["converged"] = report.coupled.converged
        out_data["sector_label_count"] = report.coupled.sector_label_count
        out_data["candidate_state_count"] = report.coupled.candidate_state_count
        out_data["coupled_K_curve"] = {
            "order": report.coupled.order_indices,
            "energies": report.coupled.energies,
            "K_PT": report.coupled.k_pt,
        }
        out_data["hermiticity_residual"] = report.coupled.hermiticity_residual
        out_data["variational_from_anchor"] = (
            report.coupled.variational_from_anchor
        )
        out_data["sector_eigenstates"] = [
            [list(label), int(idx)] for label, idx in report.coupled.chosen
        ]
    # Overwrite a partial file left by an interrupted stage so ``--resume``
    # always produces one valid JSON document.
    with open(outname, "w") as fp:
        json.dump(out_data, fp, indent=2)
    print("results written to", outname)


def solve_tapered_task(data):
    """Pickle-friendly worker for one Clifford-tapered sector."""
    return solve_tapered_sector(
        data["frame"],
        tuple(data["label"]),
        data["physical_indices"],
        data["n_roots"],
    )


def load_clifford_symmetries(args, input_data, moldata):
    """Load ordered Pauli symmetries from a manifest or legacy parity matrix."""
    manifest_path = args.symmetry_manifest or input_data.get("symmetry_manifest")
    if manifest_path:
        manifest = load_symmetry_manifest(manifest_path)
        return manifest["symmetries"], manifest["parity_matrix"], manifest_path

    parity_path = input_data.get("parity")
    if parity_path is None:
        raise ValueError("Clifford backend needs a symmetry manifest or parity matrix")
    parity_matrix = np.atleast_2d(np.loadtxt(parity_path, dtype=int))
    symmetries = z_symmetries_from_parity_matrix(parity_matrix, moldata.norb)
    return symmetries, parity_matrix, None


def solve_clifford_sectors(frame, physical_sectors, labels, n_roots, parallel):
    """Solve requested tapered sectors serially or with mpi4py futures."""
    solve_frame = {
        "hamiltonian": frame["hamiltonian"],
        "n_symmetries": frame["n_symmetries"],
        "n_residual_qubits": frame["n_residual_qubits"],
    }
    tasks = [
        {
            "frame": solve_frame,
            "label": label,
            "physical_indices": physical_sectors[label],
            "n_roots": n_roots,
        }
        for label in labels
    ]
    results = {}
    if parallel:
        with MPIPoolExecutor() as executor:
            iterator = executor.map(solve_tapered_task, tasks)
            for result in iterator:
                results[tuple(result["label"])] = result
    else:
        for task in tasks:
            result = solve_tapered_task(task)
            results[tuple(result["label"])] = result
    return results


def solve_physical_clifford_sectors(physical_matrix, physical_basis, labels, n_roots):
    """Solve sector blocks by slicing one physical Clifford-frame matrix."""
    results = {}
    for label in labels:
        results[label] = solve_physical_clifford_sector(
            physical_matrix,
            label,
            physical_basis["residual_indices"][label],
            physical_basis["physical_positions"][label],
            n_roots,
        )
    return results


def build_tapered_block_task(data):
    """Build one off-diagonal tapered block in a worker-safe form."""
    operator = tapered_operator(
        data["frame"], tuple(data["bra_label"]), tuple(data["ket_label"])
    )
    matrix = restricted_operator_matrix(
        operator,
        data["frame"]["n_residual_qubits"],
        data["bra_indices"],
        data["ket_indices"],
    )
    return {
        "key": (tuple(data["bra_label"]), tuple(data["ket_label"])),
        "matrix": matrix,
    }


def build_coupled_block_cache(frame, sector_results, candidates, parallel):
    """Reuse diagonal blocks and build each distinct off-diagonal block once."""
    cache = {}
    labels = sorted({tuple(candidate["label"]) for candidate in candidates})
    for label in labels:
        cache[(label, label)] = sector_results[label]["matrix"]

    solve_frame = {
        "hamiltonian": frame["hamiltonian"],
        "n_symmetries": frame["n_symmetries"],
        "n_residual_qubits": frame["n_residual_qubits"],
    }
    tasks = []
    for bra_index, bra_label in enumerate(labels):
        for ket_label in labels[bra_index + 1:]:
            tasks.append(
                {
                    "frame": solve_frame,
                    "bra_label": bra_label,
                    "ket_label": ket_label,
                    "bra_indices": sector_results[bra_label]["physical_indices"],
                    "ket_indices": sector_results[ket_label]["physical_indices"],
                }
            )

    if parallel and tasks:
        with MPIPoolExecutor() as executor:
            for result in executor.map(build_tapered_block_task, tasks):
                cache[result["key"]] = result["matrix"]
    else:
        for task in tasks:
            result = build_tapered_block_task(task)
            cache[result["key"]] = result["matrix"]
    return cache


def build_physical_coupled_block_cache(physical_matrix, sector_results, candidates):
    """Slice coupled blocks from one physical Clifford-frame matrix."""
    cache = {}
    labels = sorted({tuple(candidate["label"]) for candidate in candidates})
    for label in labels:
        cache[(label, label)] = sector_results[label]["matrix"]

    for bra_index, bra_label in enumerate(labels):
        bra_positions = np.asarray(
            sector_results[bra_label]["physical_positions"], dtype=int
        )
        for ket_label in labels[bra_index + 1:]:
            ket_positions = np.asarray(
                sector_results[ket_label]["physical_positions"], dtype=int
            )
            cache[(bra_label, ket_label)] = physical_matrix[bra_positions, :][
                :, ket_positions
            ]
    return cache


def sector_result_metadata(sector_results, frame):
    """Return JSON-safe sector diagnostics without expanding Pauli matrices."""
    return [
        {
            "label": list(label),
            "dimension": result["dimension"],
            "energies": [float(np.real(value)) for value in result["energies"]],
            "solver": result["solver"],
            "pauli_count": result.get("pauli_count"),
            "lcu_one_norm": result.get("lcu_one_norm"),
            "hermitian": (
                pauli_lcu_is_hermitian(
                    result["operator"], frame["n_residual_qubits"]
                )
                if "operator" in result
                else None
            ),
        }
        for label, result in sorted(sector_results.items())
    ]


def save_tapered_lcus(path, frame, sector_results, block_labels):
    """Save diagonal and required off-diagonal tapered Pauli LCUs."""
    diagonal = []
    for label in sorted(sector_results):
        diagonal.append(
            {
                "label": list(label),
                "operator": qubit_operator_to_data(sector_results[label]["operator"]),
            }
        )

    off_diagonal = []
    for bra_label, ket_label in sorted(block_labels):
        if bra_label == ket_label:
            continue
        operator = tapered_operator(frame, bra_label, ket_label)
        if operator.terms:
            off_diagonal.append(
                {
                    "bra_label": list(bra_label),
                    "ket_label": list(ket_label),
                    "operator": qubit_operator_to_data(operator),
                }
            )

    data = {
        "schema": "quasisymmetry.tapered_lcu",
        "version": 1,
        "hermitian_conjugate_blocks_implicit": True,
        "n_parent_qubits": frame["n_qubits"],
        "n_tapered_qubits": frame["n_residual_qubits"],
        "n_symmetries": frame["n_symmetries"],
        "diagonal": diagonal,
        "off_diagonal": off_diagonal,
    }
    with Path(path).open("w") as file:
        json.dump(data, file, indent=2)


def run_clifford_metrics(args, input_data, out_data):
    """Run decoupled and coupled metrics using tapered Pauli LCUs."""
    start = time.time()
    timings = {}

    stage_start = time.time()
    moldata = load_moldata(input_data["molpath"])
    dumpdata = fcidump_data(input_data["molpath"])
    symmetries, parity_matrix, manifest_path = load_clifford_symmetries(
        args, input_data, moldata
    )
    timings["load_input"] = time.time() - stage_start

    stage_start = time.time()
    rotation_parameters = np.asarray(input_data["rotation"], dtype=float)
    from src.orbital_rotation import pairs_from_oo_data

    rotation = x_to_rotation(
        rotation_parameters,
        moldata.norb,
        pairs_from_oo_data(input_data, moldata.norb),
    )
    rotated_hamiltonian = moldata.hamiltonian.rotated(rotation)
    jw_hamiltonian = molecular_hamiltonian_to_jw(rotated_hamiltonian, moldata.nelec)
    frame = build_clifford_frame(jw_hamiltonian, symmetries, 2 * moldata.norb)
    physical_basis = physical_clifford_basis(
        moldata.norb,
        moldata.nelec,
        frame["clifford"],
        frame["n_symmetries"],
    )
    physical_sectors = physical_basis["residual_indices"]
    timings["build_clifford_frame"] = time.time() - stage_start

    requested_labels = parse_sector_labels(args.sector_labels, frame["n_symmetries"])
    labels = sorted(physical_sectors) if requested_labels is None else requested_labels
    missing = [label for label in labels if label not in physical_sectors]
    if missing:
        raise ValueError(f"requested sector labels have no physical determinants: {missing}")

    n_roots = args.n_roots if args.n_roots is not None else args.states_per_sector
    stage_start = time.time()
    physical_matrix = None
    if args.clifford_block_builder == "physical":
        physical_matrix = physical_clifford_matrix(frame, physical_basis["full_indices"])
        timings["build_physical_clifford_matrix"] = time.time() - stage_start
        stage_start = time.time()
        sector_results = solve_physical_clifford_sectors(
            physical_matrix,
            physical_basis,
            labels,
            n_roots,
        )
    else:
        sector_results = solve_clifford_sectors(
            frame,
            physical_sectors,
            labels,
            n_roots,
            args.parallel_sectors,
        )
    timings["solve_sectors"] = time.time() - stage_start

    stage_start = time.time()
    exact_energy, _ = get_fci(dumpdata)
    timings["solve_parent_fci"] = time.time() - stage_start

    decoupled_energy = min(
        float(result["energies"][0]) for result in sector_results.values()
    )
    candidates = []
    block_cache = {}
    reference_weights = np.asarray([])
    curve = {"order": [], "energies": [], "K": None, "converged": False}
    selected_candidates = []
    selected_sectors = []

    if not args.decoupled_only:
        candidates = sector_state_candidates(sector_results)
        stage_start = time.time()
        if args.clifford_block_builder == "physical":
            block_cache = build_physical_coupled_block_cache(
                physical_matrix,
                sector_results,
                candidates,
            )
        else:
            block_cache = build_coupled_block_cache(
                frame,
                sector_results,
                candidates,
                args.parallel_coupled_blocks,
            )
        timings["build_coupled_blocks"] = time.time() - stage_start

        stage_start = time.time()
        h_coupled, _ = candidate_hamiltonian(frame, candidates, block_cache)
        timings["assemble_coupled_hamiltonian"] = time.time() - stage_start

        stage_start = time.time()
        if args.coupled_energy_method == "reference":
            from src.dmrg_solver import DMRGConfig, get_dmrg_reference

            _, overlap_vec = get_dmrg_reference(
                dumpdata,
                store_dir=args.wavefunction_dir,
                config=DMRGConfig(max_bond_dim=args.bond_dim),
                n_threads=args.n_threads,
                reuse=True,
            )
            rotated_overlap = ffsim.apply_orbital_rotation(
                overlap_vec,
                rotation,
                norb=moldata.norb,
                nelec=moldata.nelec,
            )
            jw_reference = ci_vector_to_jw_state(
                rotated_overlap,
                moldata.norb,
                moldata.nelec,
            )
            transformed_reference = frame["clifford"].transform_state(jw_reference)
            reference_weights = candidate_reference_weights(
                frame,
                candidates,
                transformed_reference,
            )
            order = reference_candidate_order(reference_weights)
            curve = coupled_energy_curve(
                h_coupled,
                order,
                exact_energy=exact_energy,
                tolerance=CHEMICAL_PRECISION,
            )
            out_data["overlap_reference"] = "dmrg"
        else:
            curve = perturbative_coupled_energy_curve(
                h_coupled,
                exact_energy=exact_energy,
                tolerance=CHEMICAL_PRECISION,
            )
        timings["select_coupled_space"] = time.time() - stage_start

        selected_count = curve["K"] if curve["K"] is not None else len(curve["order"])
        selected_candidate_indices = curve["order"][:selected_count]
        selected_candidates = [candidates[index] for index in selected_candidate_indices]
        selected_sectors = sorted(
            set(candidate["label"] for candidate in selected_candidates)
        )

    stage_start = time.time()
    serialized_sector_results = sector_result_metadata(sector_results, frame)
    timings["collect_sector_metadata"] = time.time() - stage_start

    out_data.update(
        {
            "sector_backend": "clifford",
            "clifford_block_builder": args.clifford_block_builder,
            "symmetry_manifest": manifest_path,
            "parity_matrix": parity_matrix.tolist(),
            "clifford": {
                "synthesis_basis": "Z",
                "generator_mapping": "positive_z",
                "factor_descriptions": list(frame["clifford"].factor_descriptions),
                "permutation": list(frame["clifford"].permutation),
            },
            "n_parent_qubits": frame["n_qubits"],
            "n_tapered_qubits": frame["n_residual_qubits"],
            "qubit_reduction": frame["n_symmetries"],
            "n_symmetries": frame["n_symmetries"],
            "sector_labels": [list(label) for label in labels],
            "parent_jw_pauli_count": len(jw_hamiltonian.terms),
            "parent_jw_lcu_one_norm": float(
                sum(abs(complex(value)) for value in jw_hamiltonian.terms.values())
            ),
            "clifford_pauli_count": len(frame["hamiltonian"].terms),
            "clifford_lcu_one_norm": float(
                sum(abs(complex(value)) for value in frame["hamiltonian"].terms.values())
            ),
            "E_FCI": float(exact_energy),
            "E_decoupled": decoupled_energy,
            "dE": decoupled_energy - float(exact_energy),
            "candidate_count": len(candidates),
            "reference_weight_sum": (
                float(np.sum(reference_weights)) if reference_weights.size else None
            ),
            "coupled_metrics_computed": not args.decoupled_only,
            "K": curve["K"],
            "converged": curve["converged"],
            "E_coupled": curve["energies"][-1] if curve["energies"] else None,
            "coupled_curve": curve,
            "sector_eigenstates": [
                [list(candidate["label"]), candidate["root"]]
                for candidate in selected_candidates
            ],
            "relevant_sectors": [list(label) for label in selected_sectors],
            "relevant_sectors_count": len(selected_sectors),
            "relevant_sectors_total_dim": sum(
                sector_results[label]["dimension"] for label in selected_sectors
            ),
            "sector_results": serialized_sector_results,
            "timings": timings,
        }
    )

    if args.save_tapered_lcu:
        if args.clifford_block_builder != "tapered":
            raise ValueError(
                "--save_tapered_lcu requires --clifford_block_builder tapered"
            )
        stage_start = time.time()
        save_tapered_lcus(
            args.save_tapered_lcu,
            frame,
            sector_results,
            block_cache.keys(),
        )
        out_data["tapered_lcu_file"] = args.save_tapered_lcu
        timings["serialize_tapered_lcu"] = time.time() - stage_start

    timings["total"] = time.time() - start
    out_data["elapsed"] = timings["total"]

    print("Clifford backend")
    print("  parent qubits:", frame["n_qubits"])
    print("  tapered qubits:", frame["n_residual_qubits"])
    print("  block builder:", args.clifford_block_builder)
    print("  physical sectors:", len(sector_results))
    print("  E_decoupled:", decoupled_energy)
    print("  K:", curve["K"])
    print("  converged:", curve["converged"])
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sector metrics (E_decoupled, K) from an OO JSON. "
            "Use --backend for the sector eigensolver; "
            "--coupled_energy_method reference uses a DMRG wavefunction for "
            "overlap ordering (PT needs no overlap reference). "
            "Rebuilds U from JSON rotation using orbital_rotation/irreps "
            "when present. See --help epilog."
        ),
    )
    parser.add_argument(
        "input_data", help="JSON you got from optimize_symmetries.py"
    )
    add_metrics_workflow_args(parser)
    parser.add_argument("--penalty", type=float, default=30.0,
                        help="sector penalty for DMRG E_decoupled / K")
    parser.add_argument(
        "--dmrg_sweeps",
        type=int,
        default=20,
        help="sweeps for parent and sector DMRG solves",
    )
    parser.add_argument("--max_sectors", type=int, default=16,
                        help="max sectors to scan in DMRG diagnostics")
    parser.add_argument("--reorder", choices=("fiedler", "gaopt"), default=None,
                        help="optional orbital reordering before DMRG")
    parser.add_argument("--entanglement", action="store_true",
                        help="with --backend dmrg, also report orbital entropies")
    parser.add_argument("--davidson_tol", type=float, default=1e-12,
                        help="Davidson convergence tol (--backend davidson)")
    parser.add_argument("--davidson_max_cycle", type=int, default=50,
                        help="Davidson max iterations (--backend davidson)")
    parser.add_argument("--davidson_max_space", type=int, default=12,
                        help="Davidson max subspace size (--backend davidson)")
    parser.add_argument("--states_per_sector", type=int, default=500)
    parser.add_argument(
        "--n_roots",
        type=int,
        default=None,
        help="roots per tapered sector; defaults to --states_per_sector",
    )
    parser.add_argument(
        "--sector_backend",
        choices=("determinant", "clifford"),
        default="determinant",
        help="sector representation for FCI/Lanczos metrics",
    )
    parser.add_argument(
        "--clifford_block_builder",
        choices=("tapered", "physical"),
        default="tapered",
        help=(
            "tapered builds one residual Pauli block per sector pair; physical "
            "builds one Clifford-frame matrix on physical determinants"
        ),
    )
    parser.add_argument(
        "--symmetry_manifest",
        default=None,
        help="ordered Z-product symmetry manifest for the Clifford backend",
    )
    parser.add_argument(
        "--sector_labels",
        default=None,
        help="comma-separated binary tapered-sector labels, for example 000,011",
    )
    parser.add_argument(
        "--parallel_sectors",
        action="store_true",
        help="solve tapered sectors through mpi4py worker processes",
    )
    parser.add_argument(
        "--parallel_coupled_blocks",
        action="store_true",
        help="build independent off-diagonal tapered blocks through mpi4py workers",
    )
    parser.add_argument(
        "--decoupled_only",
        action="store_true",
        help="stop after diagonal sector blocks and the decoupled-energy metric",
    )
    parser.add_argument(
        "--save_tapered_lcu",
        default=None,
        help="write diagonal and needed off-diagonal tapered Pauli LCUs to JSON",
    )
    parser.add_argument("--outname", default=None,
                        help="output JSON path")
    parser.add_argument("--check_if_enough", action="store_true")
    parser.add_argument(
        "--coupled_energy_method",
        choices=("reference", "perturbation"),
        default="perturbation",
        help="K selection on CI backends: perturbation=one-shot PT (default, "
             "no overlap wavefunction); reference=overlap ordering vs a DMRG "
             "wavefunction. --backend dmrg always uses PT.",
    )
    args = parser.parse_args()
    print_workflow_banner(
        "metrics",
        backend=args.backend,
        bond_dim=(
            args.bond_dim
            if args.backend == "dmrg" or args.coupled_energy_method == "reference"
            else None
        ),
        sector_backend=args.sector_backend,
        coupled_energy_method=(
            args.coupled_energy_method if args.backend != "dmrg" else "perturbation"
        ),
    )

    with open(args.input_data, "r") as fp:
        input_data = json.load(fp)

    p = Path(input_data["molpath"])
    outname = args.outname or (
        "metrics_" + p.parts[-1] + "_" + str(uuid4())[:8] + ".json"
    )
    out_data = {"args": vars(args), "OO_data": input_data}

    if args.sector_backend == "clifford":
        if args.backend != "fci":
            parser.error(
                "--sector_backend clifford currently requires --backend fci "
                "(determinant eigsh/eigh path)"
            )
        run_clifford_metrics(args, input_data, out_data)
        with open(outname, "w") as fp:
            json.dump(out_data, fp, indent=2)
        print("Saved metrics to", outname)
        raise SystemExit(0)

    if args.backend == "dmrg":
        _run_dmrg_from_oo_json(input_data, args, outname, out_data)
        raise SystemExit(0)

    moldata = load_moldata(input_data["molpath"])
    dumpdata = fcidump_data(input_data["molpath"])

    parity_matrix = np.loadtxt(input_data["parity"], dtype=int)
    symmetries = parity_matrix_to_quasisymmetries(
        parity_matrix, moldata.norb, moldata.nelec
    )

    print(parity_matrix)

    sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)

    x = np.array(input_data["rotation"])
    from src.orbital_rotation import pairs_from_oo_data

    U = x_to_rotation(x, moldata.norb, pairs_from_oo_data(input_data, moldata.norb))

    rotated_h = moldata.hamiltonian.rotated(U)
    rotated_h_linop = ffsim.linear_operator(
        rotated_h, norb=moldata.norb, nelec=moldata.nelec
    )

    e_ref, _ = get_fci(dumpdata)
    print("FCI ", e_ref)

    out_data["backend"] = args.backend
    out_data["solver"] = args.backend  # alias for older consumers
    out_data["E_FCI"] = e_ref

    print("qty of sectors ", len(sectors.keys()))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("rank and size", rank, size)

    use_davidson = args.backend == "davidson"

    tasks = [
        {
            "moldata": moldata,
            "rotated_h": rotated_h,
            "states_per_sector": args.states_per_sector,
            "sector_label": k,
            "sector_bitstrings": v,
            **(
                {
                    "davidson_tol": args.davidson_tol,
                    "davidson_max_cycle": args.davidson_max_cycle,
                    "davidson_max_space": args.davidson_max_space,
                }
                if use_davidson
                else {}
            ),
        }
        for k, v in sectors.items()
    ]

    worker = solve_eigs_davidson if use_davidson else solve_eigs
    sector_eigs = {}
    sector_solver_meta = {}
    solve_start = time.perf_counter()
    with MPIPoolExecutor() as executor:
        for r in executor.map(worker, tasks):
            label = tuple(r["sector_label"])
            sector_eigs[label] = r["sector_eigs"]
            if "meta" in r:
                sector_solver_meta[str(list(label))] = r["meta"]
    solve_time_s = time.perf_counter() - solve_start
    out_data["solve_time_s"] = float(solve_time_s)
    if sector_solver_meta:
        out_data["sector_solver_meta"] = sector_solver_meta
        out_data["davidson_all_converged"] = all(
            meta.get("converged", False) for meta in sector_solver_meta.values()
        )
    print("sector solve_time_s", solve_time_s)

    sector_gs_energies = []
    for w, v in sector_eigs.items():
        sector_gs_energies.append(np.min(v[0]))

    smallest = np.min(sector_gs_energies)

    de_dec = smallest - e_ref
    print("Decoupled error ", de_dec)
    out_data["E_decoupled"] = smallest
    out_data["dE"] = de_dec

    h_apply = lambda v: rotated_h_linop @ v

    if args.coupled_energy_method == "perturbation":
        print("Calculating K via one-shot PT ordering + nested variational search")
        sector_data = sector_data_from_gs_pairs(
            sectors, sector_eigs, rotated_h_linop.shape[0]
        )
        e_coupled, k_coupled, converged, chosen_keys = coupled_energy_perturbation(
            h_apply,
            sector_data,
            e_exact=e_ref,
            tol=CHEMICAL_PRECISION,
        )
        print("E_coupled", e_coupled)
        print("K", k_coupled)
        print("converged", converged)
        out_data["E_coupled"] = e_coupled
        out_data["K"] = k_coupled
        out_data["converged"] = converged
        if not converged:
            print("PT coupled-energy did not converge within chemical precision")

    elif args.coupled_energy_method == "reference":
        print("Calculating K via overlap ordering against DMRG wavefunction")
        from src.dmrg_solver import DMRGConfig, get_dmrg_reference

        _, refvec = get_dmrg_reference(
            dumpdata,
            store_dir=args.wavefunction_dir,
            config=DMRGConfig(max_bond_dim=args.bond_dim),
            n_threads=args.n_threads,
            reuse=True,
        )
        rotated_refvec = ffsim.apply_orbital_rotation(
            refvec, U, norb=moldata.norb, nelec=moldata.nelec
        )
        out_data["overlap_reference"] = "dmrg"

        full_space_vectors = []
        for k, v in sectors.items():
            full_space_vectors_in_sector = np.zeros(
                (rotated_h_linop.shape[0], sector_eigs[k][0].shape[0]),
                dtype="complex",
            )
            full_space_vectors_in_sector[v, :] = sector_eigs[k][1]
            full_space_vectors.append(full_space_vectors_in_sector)
        full_space_vectors_cat = np.concatenate(full_space_vectors, axis=1)

        k_min, e_coupled, converged, weights_order = reference_coupled_energy_k(
            h_apply,
            full_space_vectors_cat,
            rotated_refvec,
            e_ref,
            chemical_precision=CHEMICAL_PRECISION,
        )
        print("E_coupled (full projection)", e_coupled)
        out_data["K"] = k_min
        if k_min is None:
            print("Not enough states per sector")
            quit()

        print("K ", k_min)

        all_state_labels = state_labels_for_columns(sector_eigs)
        chosen_keys = [all_state_labels[weights_order[i]] for i in range(k_min)]

    print("Sector eigenstates used (sector and excitation level):")
    for key in chosen_keys:
        print(key)
    out_data["sector_eigenstates"] = chosen_keys

    unique_sectors_used = list({w[0] for w in chosen_keys})
    total_dim_of_relevant_sectors = 0
    print("Relevant sectors and their dimensions:")
    for s in unique_sectors_used:
        print(s, len(sectors[s]))
        total_dim_of_relevant_sectors += len(sectors[s])
    print("{0:} sectors in total".format(len(unique_sectors_used)))
    print("Total dimension: {0:}".format(total_dim_of_relevant_sectors))

    out_data["relevant_sectors"] = unique_sectors_used
    out_data["relevant_sectors_count"] = len(unique_sectors_used)
    out_data["relevant_sectors_total_dim"] = total_dim_of_relevant_sectors
    with open(outname, "a") as fp:
        json.dump(out_data, fp, indent=2)
