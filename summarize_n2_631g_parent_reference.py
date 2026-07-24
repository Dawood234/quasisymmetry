#!/usr/bin/env python3
"""Compare independent N2/6-31G parent DMRG reference calculations."""

import argparse
import json
from pathlib import Path


def parse_args():
    """Read the parent-reference directory and convergence tolerance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--bond_dims", default="350,500")
    parser.add_argument("--tolerance_mha", type=float, default=0.2)
    return parser.parse_args()


def read_energy(path):
    """Read the energy from one ``solve_dmrg.py`` result."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("E_DMRG "):
            return float(line.split()[1])
    raise ValueError(f"E_DMRG was not found in {path}")


def main():
    """Print and save the bond-dimension convergence comparison."""
    args = parse_args()
    root = args.reference_root.resolve()
    bond_dims = [int(value) for value in args.bond_dims.split(",")]
    energies = {}

    for bond_dim in bond_dims:
        path = root / f"M{bond_dim}" / f"parent_M{bond_dim}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        energies[bond_dim] = read_energy(path)

    lower = bond_dims[-2]
    upper = bond_dims[-1]
    difference_mha = 1000.0 * abs(energies[upper] - energies[lower])
    converged = difference_mha <= args.tolerance_mha
    result = {
        "system": "N2",
        "basis": "6-31G",
        "energies_Ha": {str(key): value for key, value in energies.items()},
        "comparison": {
            "lower_bond_dimension": lower,
            "upper_bond_dimension": upper,
            "absolute_difference_mHa": difference_mha,
            "tolerance_mHa": float(args.tolerance_mha),
            "converged": bool(converged),
        },
    }
    output = root / "parent_reference_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for bond_dim in bond_dims:
        print(f"M={bond_dim}: {energies[bond_dim]:.12f} Ha")
    print(f"|E_M{upper} - E_M{lower}| = {difference_mha:.6f} mHa")
    print(f"converged within {args.tolerance_mha:.3f} mHa: {converged}")
    print("summary:", output)


if __name__ == "__main__":
    main()
