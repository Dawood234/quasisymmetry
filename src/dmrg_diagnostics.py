"""MPS-native energy-sector and entanglement diagnostics.

Phase-3 replacement for the dense ``metrics.subspace_matrix`` path:

* ``E_decoupled`` from sector-targeted DMRG on the exactly decoupled
  Hamiltonian (see :meth:`Block2DMRGSolver.sector_ground_state`);
* coupled-energy ``K`` via one-shot PT ordering + nested variational search
  over DMRG sector eigenstates, using ``<phi_i|H|phi_j>`` MPO expectations;
* orbital entropies and mutual information from the stored ground-state MPS.

Parity matrices given in the *original* orbital order are remapped through
``solver.remap_parity_matrix`` when the solver was built with Fiedler (or
other) orbital reordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.coupled_energy_core import (
    CHEMICAL_PRECISION,
    COUPLED_ENERGY_DEGENERACY_FLOOR,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TAU_PT,
    one_shot_from_hamiltonian,
)
from src.dmrg_solver import Block2DMRGSolver, DMRGConfig

logger = logging.getLogger(__name__)


@dataclass
class SectorState:
    """One DMRG eigenstate of a parity sector."""

    energy: float
    sector_label: tuple[int, ...]
    block_index: int
    mps_tag: str


@dataclass
class DecoupledDiagnostic:
    e_reference: float
    e_decoupled: float
    best_sector: tuple[int, ...]
    sector_energies: dict[tuple[int, ...], float]
    dE: float
    k_equals_one: bool


@dataclass
class CoupledDiagnostic:
    e_coupled: float | None
    k: int
    converged: bool
    chosen: list[tuple[tuple[int, ...], int]]
    e_reference: float
    e_decoupled: float
    sector_label_count: int = 0
    candidate_state_count: int = 0
    order_indices: list[int] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    k_pt: int | None = None
    hermiticity_residual: float = 0.0
    variational_from_anchor: bool = False


@dataclass
class EntanglementDiagnostic:
    bipartite: np.ndarray
    orbital_s1: np.ndarray
    mutual_information: np.ndarray


@dataclass
class DMRGMetricsReport:
    e_reference: float
    decoupled: DecoupledDiagnostic
    coupled: CoupledDiagnostic | None = None
    entanglement: EntanglementDiagnostic | None = None
    symmetry_expectations: np.ndarray | None = None
    orbital_permutation: tuple[int, ...] = field(default_factory=tuple)


def prepare_parity_matrix(
    solver: Block2DMRGSolver, parity_matrix: np.ndarray
) -> np.ndarray:
    """Parity matrix in the solver's (possibly reordered) orbital basis."""
    return solver.remap_parity_matrix(np.atleast_2d(np.asarray(parity_matrix, dtype=int)))


def decoupled_energy_dmrg(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    e_reference: float,
    *,
    config: DMRGConfig | None = None,
    penalty: float = 30.0,
    max_sectors: int = 16,
    chemical_precision: float = CHEMICAL_PRECISION,
    reference_tag: str = "GS",
) -> DecoupledDiagnostic:
    """Scan dominant sectors and return the lowest single-sector energy."""
    parity = prepare_parity_matrix(solver, parity_matrix)
    config = config or DMRGConfig()
    labels = solver.dominant_sector_labels(
        parity,
        ket=solver.get_mps(reference_tag),
        max_sectors=max_sectors,
    )
    if not labels:
        raise RuntimeError("no sector weight found in the reference MPS")

    sector_energies: dict[tuple[int, ...], float] = {}
    best_energy, best_label = np.inf, labels[0][0]
    for label, weight in labels:
        result = solver.sector_ground_state(
            parity, label, penalty=penalty, config=config
        )
        sector_energies[label] = result.energy
        logger.info(
            "sector %s weight=%.4f E=%.10f", label, weight, result.energy
        )
        if result.energy < best_energy:
            best_energy, best_label = result.energy, label

    dE = best_energy - e_reference
    k_equals_one = abs(dE) < chemical_precision
    return DecoupledDiagnostic(
        e_reference=e_reference,
        e_decoupled=best_energy,
        best_sector=best_label,
        sector_energies=sector_energies,
        dE=dE,
        k_equals_one=k_equals_one,
    )


def collect_sector_states(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    sector_labels: Sequence[tuple[int, ...]],
    *,
    nroots: int = 5,
    penalty: float = 30.0,
    config: DMRGConfig | None = None,
) -> list[SectorState]:
    """DMRG eigenstates for each requested sector, sorted by energy."""
    parity = prepare_parity_matrix(solver, parity_matrix)
    config = config or DMRGConfig()
    states: list[SectorState] = []
    for label in sector_labels:
        roots = solver.sector_excited_states(
            parity, label, nroots=nroots, penalty=penalty, config=config
        )
        for block_index, (energy, tag) in enumerate(roots):
            states.append(SectorState(
                energy=energy,
                sector_label=tuple(label),
                block_index=block_index,
                mps_tag=tag,
            ))
    states.sort(key=lambda s: s.energy)
    return states


def _hamiltonian_element(
    solver: Block2DMRGSolver, bra_tag: str, ket_tag: str
) -> complex:
    bra = solver.get_mps(bra_tag)
    ket = solver.get_mps(ket_tag) if ket_tag != bra_tag else bra
    return complex(solver.expectation(solver.hamiltonian_mpo(), ket=ket, bra=bra))


def build_mps_candidate_hamiltonian(
    solver: Block2DMRGSolver,
    states: Sequence[SectorState],
) -> np.ndarray:
    """Dense candidate Hamiltonian from MPO expectations."""
    n = len(states)
    h_mat = np.zeros((n, n), dtype=np.complex128)
    for j, ket in enumerate(states):
        for i, bra in enumerate(states[: j + 1]):
            if i == j:
                value = complex(ket.energy)
            else:
                value = _hamiltonian_element(solver, bra.mps_tag, ket.mps_tag)
            h_mat[i, j] = value
            h_mat[j, i] = np.conjugate(value)
    return 0.5 * (h_mat + h_mat.conj().T)


def one_shot_coupled_energy_mps(
    solver: Block2DMRGSolver,
    states: Sequence[SectorState],
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    tau_pt: float = DEFAULT_TAU_PT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
    max_total_vectors: int | None = None,
) -> tuple[float | None, int, bool, list[tuple[tuple[int, ...], int]]]:
    """One-shot PT + nested variational ``K`` using MPS Hamiltonian elements."""
    if not states:
        return None, 0, False, []
    h_coupled = build_mps_candidate_hamiltonian(solver, states)
    keys = [(s.sector_label, s.block_index) for s in states]
    result = one_shot_from_hamiltonian(
        h_coupled,
        e_exact=e_exact,
        tol=tol,
        tau_pt=tau_pt,
        block_size=block_size,
        degeneracy_floor=degeneracy_floor,
        keys=keys,
        max_total_vectors=max_total_vectors,
    )
    return result.as_tuple()


def one_shot_coupled_result_mps(
    solver: Block2DMRGSolver,
    states: Sequence[SectorState],
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    tau_pt: float = DEFAULT_TAU_PT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
    max_total_vectors: int | None = None,
):
    """Return the complete one-shot PT result and candidate Hamiltonian."""
    if not states:
        return None, np.zeros((0, 0), dtype=np.complex128)
    h_coupled = build_mps_candidate_hamiltonian(solver, states)
    keys = [(state.sector_label, state.block_index) for state in states]
    result = one_shot_from_hamiltonian(
        h_coupled,
        e_exact=e_exact,
        tol=tol,
        tau_pt=tau_pt,
        block_size=block_size,
        degeneracy_floor=degeneracy_floor,
        keys=keys,
        max_total_vectors=max_total_vectors,
    )
    return result, h_coupled


def coupled_energy_dmrg(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    e_reference: float,
    e_decoupled: float,
    *,
    sector_labels: Sequence[tuple[int, ...]] | None = None,
    nroots: int = 5,
    penalty: float = 30.0,
    config: DMRGConfig | None = None,
    max_sectors: int = 8,
    chemical_precision: float = CHEMICAL_PRECISION,
) -> CoupledDiagnostic:
    """Build sector spectra with DMRG and run the one-shot PT ``K`` diagnostic.

    ``parity_matrix`` must be in the *original* orbital order; remapping for a
    reordered solver is handled inside :func:`collect_sector_states`.
    """
    if sector_labels is None:
        remapped = prepare_parity_matrix(solver, parity_matrix)
        sector_labels = [
            label for label, _ in solver.dominant_sector_labels(
                remapped, max_sectors=max_sectors
            )
        ]
    states = collect_sector_states(
        solver,
        parity_matrix,
        sector_labels,
        nroots=nroots,
        penalty=penalty,
        config=config,
    )
    result, h_coupled = one_shot_coupled_result_mps(
        solver,
        states,
        e_exact=e_reference,
        tol=chemical_precision,
    )
    if result is None:
        e_coupled, k, converged, chosen = None, 0, False, []
        order_indices, energies, k_pt = [], [], None
    else:
        e_coupled, k, converged, chosen = result.as_tuple()
        order_indices = [int(index) for index in result.order_indices]
        energies = [float(energy) for energy in result.energies]
        k_pt = None if result.K_pt is None else int(result.K_pt)
    hermiticity = float(np.max(np.abs(h_coupled - h_coupled.conj().T))) if h_coupled.size else 0.0
    variational = (
        e_coupled is not None and float(e_coupled) <= float(e_decoupled) + 1.0e-9
    )
    return CoupledDiagnostic(
        e_coupled=e_coupled,
        k=k,
        converged=converged,
        chosen=chosen,
        e_reference=e_reference,
        e_decoupled=e_decoupled,
        sector_label_count=len(sector_labels),
        candidate_state_count=len(states),
        order_indices=order_indices,
        energies=energies,
        k_pt=k_pt,
        hermiticity_residual=hermiticity,
        variational_from_anchor=bool(variational),
    )


def entanglement_diagnostic(
    solver: Block2DMRGSolver, ket=None
) -> EntanglementDiagnostic:
    """Bipartite cut entropies, 1-orbital entropies and mutual information."""
    if ket is None:
        ket = solver.get_mps()
    return EntanglementDiagnostic(
        bipartite=solver.bipartite_entanglement(ket=ket),
        orbital_s1=solver.orbital_entropies(ket=ket, orb_type=1),
        mutual_information=solver.mutual_information(ket=ket),
    )


def run_dmrg_metrics(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    *,
    config: DMRGConfig | None = None,
    penalty: float = 30.0,
    max_sectors: int = 16,
    states_per_sector: int = 5,
    compute_k: bool = True,
    compute_entanglement: bool = True,
    chemical_precision: float = CHEMICAL_PRECISION,
    reuse_ground_state: bool = True,
) -> DMRGMetricsReport:
    """Full MPS-native diagnostics: E_ref, E_dec, optional K and entropies."""
    from src.dmrg_solver import solve_or_load_ground_state

    config = config or DMRGConfig()
    gs = solve_or_load_ground_state(
        solver, config=config, reuse=reuse_ground_state
    )
    parity = prepare_parity_matrix(solver, parity_matrix)
    reference_mps = solver.get_mps(gs.mps_tag)
    expectations = solver.symmetry_expectations(parity, ket=reference_mps)

    decoupled = decoupled_energy_dmrg(
        solver,
        parity_matrix,  # original order; remapped inside
        gs.energy,
        config=config,
        penalty=penalty,
        max_sectors=max_sectors,
        chemical_precision=chemical_precision,
        reference_tag=gs.mps_tag,
    )

    # Unconstrained DMRG can trap in an excited near-parity sector when the
    # quasi-symmetries are nearly exact. Sector scan measures bare <H>, so a
    # lower sector energy is a strictly better variational reference.
    e_ref = float(gs.energy)
    ref_ket = solver.get_mps(gs.mps_tag)
    promoted = False
    if decoupled.e_decoupled < e_ref - chemical_precision:
        promoted = True
        e_ref = float(decoupled.e_decoupled)
        best_tag = "SECTOR_" + "".join(map(str, decoupled.best_sector))
        ref_ket = solver.get_mps(best_tag)
        decoupled = DecoupledDiagnostic(
            e_reference=e_ref,
            e_decoupled=decoupled.e_decoupled,
            best_sector=decoupled.best_sector,
            sector_energies=decoupled.sector_energies,
            dE=0.0,
            k_equals_one=True,
        )

    if promoted:
        expectations = solver.symmetry_expectations(parity, ket=ref_ket)

    coupled = None
    if compute_k and not decoupled.k_equals_one:
        coupled = coupled_energy_dmrg(
            solver,
            parity_matrix,
            e_ref,
            decoupled.e_decoupled,
            sector_labels=list(decoupled.sector_energies.keys()),
            nroots=states_per_sector,
            penalty=penalty,
            config=config,
            chemical_precision=chemical_precision,
        )
    elif decoupled.k_equals_one:
        coupled = CoupledDiagnostic(
            e_coupled=decoupled.e_decoupled,
            k=1,
            converged=True,
            chosen=[(decoupled.best_sector, 0)],
            e_reference=e_ref,
            e_decoupled=decoupled.e_decoupled,
            sector_label_count=len(decoupled.sector_energies),
            candidate_state_count=1,
            order_indices=[0],
            energies=[float(decoupled.e_decoupled)],
            k_pt=1,
            variational_from_anchor=True,
        )

    ent = (
        entanglement_diagnostic(solver, ket=ref_ket)
        if compute_entanglement
        else None
    )
    return DMRGMetricsReport(
        e_reference=e_ref,
        decoupled=decoupled,
        coupled=coupled,
        entanglement=ent,
        symmetry_expectations=expectations,
        orbital_permutation=tuple(solver.orbital_permutation),
    )


def format_metrics_report(report: DMRGMetricsReport) -> list[str]:
    """Plain-text lines matching the style of ``metrics.py`` result files."""
    lines = [
        f"solver dmrg",
        f"orbital_permutation {list(report.orbital_permutation)}",
        f"E_FCI {report.e_reference:4.6f}",
        f"E_decoupled {report.decoupled.e_decoupled:4.6f}",
        f"dE {report.decoupled.dE:4.6f}",
        f"best_sector {report.decoupled.best_sector}",
    ]
    if report.symmetry_expectations is not None:
        lines.append(
            f"symmetry expectations {np.round(report.symmetry_expectations, 6)}"
        )
    if report.coupled is not None:
        lines.append("coupled_energy_method one_shot_perturbation")
        if report.coupled.e_coupled is not None:
            lines.append(f"E_coupled {report.coupled.e_coupled:4.6f}")
        lines.append(f"K {report.coupled.k}")
        lines.append(f"converged {report.coupled.converged}")
        lines.append(f"sector_label_count {report.coupled.sector_label_count}")
        lines.append(f"candidate_state_count {report.coupled.candidate_state_count}")
        lines.append(f"K_PT {report.coupled.k_pt}")
        lines.append(
            f"hermiticity_residual {report.coupled.hermiticity_residual:.3e}"
        )
        lines.append(
            f"variational_from_anchor {report.coupled.variational_from_anchor}"
        )
        for key in report.coupled.chosen:
            lines.append(str(key))
    if report.entanglement is not None:
        lines.append(
            f"bipartite entanglement {np.round(report.entanglement.bipartite, 6)}"
        )
        lines.append(
            f"orbital entropies {np.round(report.entanglement.orbital_s1, 6)}"
        )
        mi = report.entanglement.mutual_information
        lines.append(f"mutual_information_max {float(np.max(mi)):.6f}")
    return lines
