import pytest

from run_n2_631g_las import validate_n2_metadata


def n2_metadata():
    return {
        "norb": 18,
        "nelec": 14,
        "n_alpha": 7,
        "n_beta": 7,
        "parent_qubits": 36,
        "fixed_spin_dimension": 1_012_766_976,
        "rotation_parameters": 24,
        "orbital_irreps": [],
    }


def test_n2_dimensions_are_accepted():
    metadata = n2_metadata()

    assert validate_n2_metadata(metadata) is metadata


def test_non_n2_dimensions_are_rejected():
    metadata = n2_metadata()
    metadata["norb"] = 13

    with pytest.raises(ValueError, match="norb=18"):
        validate_n2_metadata(metadata)


def test_empty_irrep_rotation_space_is_rejected():
    metadata = n2_metadata()
    metadata["rotation_parameters"] = 0

    with pytest.raises(ValueError, match="rotation_parameters=24"):
        validate_n2_metadata(metadata)
