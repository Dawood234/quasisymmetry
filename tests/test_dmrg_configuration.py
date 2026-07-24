"""Small tests for DMRG schedules and symmetry bookkeeping."""

import numpy as np

from src.dmrg_solver import (
    DMRGConfig,
    find_sector_determinant,
    permute_orbital_symmetries,
    rotation_preserves_orbital_symmetries,
)


def test_default_schedule_reaches_requested_bond_dimension():
    config = DMRGConfig(max_bond_dim=500, n_sweeps=20)
    bond_dims, noises, thresholds = config.schedule()

    assert bond_dims == [125] * 4 + [250] * 4 + [500] * 12
    assert len(noises) == 20
    assert thresholds == [1.0e-10] * 20


def test_explicit_schedule_is_padded_to_number_of_sweeps():
    config = DMRGConfig(
        max_bond_dim=500,
        n_sweeps=5,
        bond_dims=(50, 100),
        noises=(1.0e-4,),
    )
    bond_dims, noises, _ = config.schedule()

    assert bond_dims == [50, 100, 100, 100, 100]
    assert noises == [1.0e-4, 0.0, 0.0, 0.0, 0.0]


def test_point_group_labels_follow_orbital_reordering():
    labels = (0, 3, 0, 2)
    permutation = (2, 0, 3, 1)

    assert permute_orbital_symmetries(labels, permutation) == (0, 0, 2, 3)


def test_rotation_preserves_irreps_only_for_block_diagonal_rotations():
    labels = (0, 0, 2)
    same_irrep_rotation = np.asarray(
        [
            [0.8, 0.6, 0.0],
            [-0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mixed_irrep_rotation = same_irrep_rotation.copy()
    mixed_irrep_rotation[0, 2] = 1.0e-3

    assert rotation_preserves_orbital_symmetries(
        same_irrep_rotation, labels
    )
    assert not rotation_preserves_orbital_symmetries(
        mixed_irrep_rotation, labels
    )


def test_sector_seed_respects_target_point_group_irrep():
    orbital_symmetries = (0, 1, 1)
    determinant = find_sector_determinant(
        parity_matrix=np.asarray([[0, 1, 0]]),
        sector_label=(1,),
        norb=3,
        n_elec=2,
        spin=0,
        orbital_symmetries=orbital_symmetries,
        target_irrep=0,
    )

    determinant_irrep = 0
    for orbital, symbol in enumerate(determinant):
        if symbol in ("a", "b"):
            determinant_irrep ^= orbital_symmetries[orbital]
    assert determinant_irrep == 0
