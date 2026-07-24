import numpy as np
import openfermion as of
from openfermionpyscf import run_pyscf
from dataclasses import dataclass

import ffsim
from pathlib import Path
import pyscf

CHEMICAL_PRECISION = 0.0016


@dataclass
class FakeMolecularData:
    n_electrons: int
    n_orbitals: int
    hamiltonian: of.InteractionOperator


def load_moldata(molpath) -> ffsim.MolecularData:
    p = Path(molpath)
    if p.suffix == ".chk":
        mol = pyscf.lib.chkfile.load_mol(molpath)
        mf = pyscf.scf.RHF(mol)
        mf.update_from_chk(molpath)
        moldata = ffsim.MolecularData.from_scf(mf)
    elif p.suffix == ".FCIDUMP" or str(p).endswith("FCIDUMP"):
        moldata = ffsim.MolecularData.from_fcidump(molpath)
    else:
        raise ValueError("hamiltonian must be a pyscf checkfile or an fcidump")
    return moldata


def fcidump_data(molpath):
    p = Path(molpath)
    if p.suffix == ".FCIDUMP" or str(p).endswith("FCIDUMP"):
        return pyscf.tools.fcidump.read(molpath,
                                            verbose=False)
    elif p.suffix == ".chk":
        dumpdata = {}
        mol = pyscf.lib.chkfile.load_mol(molpath)
        scf_data = pyscf.lib.chkfile.load(molpath, "scf")

        scf = pyscf.scf.RHF(mol)
        scf.update_from_chk(molpath)

        dumpdata["NORB"] = mol.nao
        dumpdata["NELEC"] = mol.nelec

        mo_coeff = scf_data["mo_coeff"]
        hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        h1 = mo_coeff.T @ hcore_ao @ mo_coeff
        dumpdata["H1"] = h1
        dumpdata["H2"] = pyscf.ao2mo.full(mol, mo_coeff)
        dumpdata["ECORE"] = mol.energy_nuc()
        if mol.symmetry and hasattr(scf, "get_orbsym"):
            dumpdata["ORBSYM"] = np.asarray(scf.get_orbsym(), dtype=int)
            dumpdata["PG_IRREP"] = int(scf.get_wfnsym())
            dumpdata["POINT_GROUP"] = str(mol.groupname)
        return dumpdata
    else:
        raise ValueError(".chk or .FCIDUMP")


def get_mol(molname, bond):
    geometry, description = get_geometry_and_description(molname, bond)

    mol = of.MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        description=description
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False, run_cisd=False)

    return mol


def build_U_from_thetas(n, thetas, pairs): #U size = n_spatial x n_spatial
    U = np.eye(n)
    for th, (p, q) in zip(thetas, pairs):
        U = U @ givens(n, p, q, th)
    return U


def givens(n, p, q, theta):
    G = np.eye(n)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    G[p, p] = c
    G[q, q] = c
    G[p, q] = s
    G[q, p] = -s
    return G


def get_model_hamiltonian():
    h = of.FermionOperator(((0, 1), (0, 0)), 1)
    h += of.FermionOperator(((1, 1), (1, 0)), 1)

    return FakeMolecularData(2, 2, of.get_interaction_operator(h, 4))


# ============================================================
# Geometry builders
# ============================================================


def build_lih_geometry(li_h_bond_angstrom: float):
    r = li_h_bond_angstrom / 2.0
    return [
        ("Li", (0.0, 0.0, -r)),
        ("H",  (0.0, 0.0, +r)),
    ]


def build_h2o_geometry(oh_bond_angstrom: float, hoh_angle_deg: float = 104.5):
    angle_rad = np.radians(hoh_angle_deg / 2.0)
    x = oh_bond_angstrom * np.sin(angle_rad)
    y = oh_bond_angstrom * np.cos(angle_rad)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (x, y, 0.0)),
        ("H", (-x, y, 0.0)),
    ]


def build_h4_linear_geometry(h_h_bond_angstrom: float):
    """
    Linear H4 chain centered at origin:
      H - H - H - H
    nearest-neighbor spacing = h_h_bond_angstrom
    """
    d = h_h_bond_angstrom
    coords = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    return [("H", (0.0, 0.0, z)) for z in coords]


def build_h4_square_geometry(side_angstrom: float):
    """
    H4 square centered at origin in the xy-plane.

    side_angstrom:
        side length of the square
    """
    s = side_angstrom / 2.0
    return [
        ("H", (-s, -s, 0.0)),
        ("H", (+s, -s, 0.0)),
        ("H", (+s, +s, 0.0)),
        ("H", (-s, +s, 0.0)),
    ]


def build_h4_rectangle_geometry(long_side_angstrom: float, aspect_ratio: float = 1.5):
    """
    H4 rectangle centered at origin in the xy-plane.

    long_side_angstrom:
        length of the longer side

    aspect_ratio:
        long_side / short_side
        must be > 0
    """
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive.")

    a = long_side_angstrom / 2.0
    b = (long_side_angstrom / aspect_ratio) / 2.0

    return [
        ("H", (-a, -b, 0.0)),
        ("H", (+a, -b, 0.0)),
        ("H", (+a, +b, 0.0)),
        ("H", (-a, +b, 0.0)),
    ]


def build_h2_geometry(bond_length: float):
    return [
        ("H", (0, 0., 0.)),
        ("H", (bond_length, 0., 0.))
    ]

def build_n2_geometry(bond_length: float):
    return [
        ("N", (0, 0., 0.)),
        ("N", (bond_length, 0., 0.))
    ]


def get_geometry_and_description(molecule: str, x: float, **kwargs):
    mol = molecule.lower()

    if mol == "lih":
        return build_lih_geometry(x), f"LiH_Bond{x:.4f}"

    elif mol == "h2o":
        angle = kwargs.get("hoh_angle_deg", 104.5)
        return build_h2o_geometry(x, hoh_angle_deg=angle), f"H2O_OH{x:.4f}_{angle:.4f}"

    elif mol == "h4_linear":
        return build_h4_linear_geometry(x), f"H4_linear_d{x:.4f}"

    elif mol == "h4_square":
        return build_h4_square_geometry(x), f"H4_square_side{x:.4f}"

    elif mol == "h4_rectangle":
        aspect_ratio = kwargs.get("aspect_ratio", 1.5)
        return build_h4_rectangle_geometry(x, aspect_ratio=aspect_ratio), f"H4_rectangle_long{x:.4f}_ar{aspect_ratio:.3f}"

    elif mol == "h2":
        return build_h2_geometry(x), f"H2_bond{x:.4f}"

    elif mol == "n2":
        return build_n2_geometry(x), f"N2_bond{x:.4f}"

    else:
        raise ValueError(
            f"Unsupported molecule '{molecule}'. "
            "Choose from: lih, h2o, h4_linear, h4_square, h4_rectangle, h2"
        )
