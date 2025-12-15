"""
Quantum Dot Dataset Generator for GaN, ZnO, and GaN/ZnO alloys.

This script generates spherical wurtzite quantum dots with pseudo-hydrogen
passivation for use in VASP band gap calculations.

================================================================================
VASP POTCAR SETUP FOR PSEUDO-HYDROGENS
================================================================================

The pseudo-hydrogen passivants require modified POTCAR files with fractional
valence charges (ZVAL). Standard H POTCAR has ZVAL=1.0, but passivating dangling
bonds requires fractional electrons based on valence electron counting.

REQUIRED FRACTIONAL CHARGES:
    - H_Ga (passivating Ga dangling bond): ZVAL = 1.25
    - H_Zn (passivating Zn dangling bond): ZVAL = 1.50
    - H_N  (passivating N dangling bond):  ZVAL = 0.75
    - H_O  (passivating O dangling bond):  ZVAL = 0.50

HOW TO CREATE MODIFIED POTCAR FILES:

Option 1: Modify ZVAL in existing H POTCAR
    1. Copy your H POTCAR: cp H/POTCAR H_1.25/POTCAR
    2. Edit the ZVAL line:
       - Find the line containing "ZVAL" (usually line 5)
       - Change "1.0000000" to "1.2500000" (or appropriate value)
    3. Repeat for each fractional charge needed

Option 2: Use VASP's built-in fractional POTCARs (if available)
    Some VASP distributions include H.25, H.5, H.75, H1.25, H1.5 POTCARs
    in the potpaw_PBE directory.

POTCAR CONCATENATION ORDER:
    For a structure with Ga, H, N, O, Zn species (sorted alphabetically by VASP),
    concatenate POTCARs in the SAME order as species appear in POSCAR:

    cat Ga/POTCAR H_modified/POTCAR N/POTCAR O/POTCAR Zn/POTCAR > POTCAR

IMPORTANT NOTES:
    - The script outputs structures with ALL H atoms using standard "H" symbol
    - You must determine which H atoms passivate which species based on their
      positions (H atoms near Ga/Zn are H_cation type, H near N/O are H_anion type)
    - For simplicity, you may use a single averaged H POTCAR if band gaps are
      not highly sensitive to passivant charge (test this assumption!)
    - The initial_charges stored in ASE are for reference only; VASP reads
      charges from POTCAR, not from the structure file

ALTERNATIVE APPROACH (simpler but less rigorous):
    Use standard H POTCAR (ZVAL=1.0) and accept small errors in the electronic
    structure near the surface. This is often acceptable for:
    - Qualitative band gap trends
    - Larger QDs where surface effects are less dominant
    - Initial screening before more careful calculations

================================================================================

Scientific basis:
- Wurtzite structure with literature lattice parameters
- Vegard's law interpolation for alloy lattice constants
- Pseudo-hydrogen passivation using fractional charges based on valence electron
  counting (dangling bond saturation model)

References:
- GaN lattice: Y. T. Tsang et al., J. Appl. Phys. 79 (1996)
- ZnO lattice: U. Ozgur et al., J. Appl. Phys. 98, 041301 (2005)
- Passivation model: X. Huang et al., Phys. Rev. B 71, 165328 (2005)

Usage:
    python gen_qdot.py --radii 8 10 12 --fractions 0.0 0.5 1.0 --seeds 42 43
    python gen_qdot.py --validate  # Validate generated structures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import read, write
from ase.neighborlist import NeighborList


# =============================================================================
# PHYSICAL CONSTANTS AND LITERATURE VALUES
# =============================================================================

GAN_LATTICE = {
    "a": 3.189,
    "c": 5.185,
    "source": "Y. T. Tsang et al., J. Appl. Phys. 79 (1996)",
}

ZNO_LATTICE = {
    "a": 3.249,
    "c": 5.206,
    "source": "U. Ozgur et al., J. Appl. Phys. 98, 041301 (2005)",
}

PASSIVANT_CHARGES = {
    "Ga": 1.25,
    "Zn": 1.50,
    "N": 0.75,
    "O": 0.50,
}

BOND_CUTOFF = 2.4
BOND_LENGTH_SCALING = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "quantum_dots"

EXPECTED_BOND_LENGTHS = {
    ("Ga", "N"): (1.94, 1.96),
    ("Zn", "O"): (1.97, 2.00),
    ("Ga", "H"): (1.40, 1.60),
    ("Zn", "H"): (1.45, 1.65),
    ("N", "H"): (1.40, 1.55),
    ("O", "H"): (1.45, 1.60),
}


# =============================================================================
# LATTICE PARAMETER FUNCTIONS
# =============================================================================

def vegards_law(fraction_zno: float) -> Tuple[float, float]:
    """Interpolate lattice constants via Vegard's law for GaN(1-x)ZnO(x)."""
    assert 0.0 <= fraction_zno <= 1.0, f"Fraction must be in [0,1], got {fraction_zno}"
    a = GAN_LATTICE["a"] + (ZNO_LATTICE["a"] - GAN_LATTICE["a"]) * fraction_zno
    c = GAN_LATTICE["c"] + (ZNO_LATTICE["c"] - GAN_LATTICE["c"]) * fraction_zno
    return a, c


# =============================================================================
# STRUCTURE BUILDING FUNCTIONS
# =============================================================================

def build_bulk_supercell(a: float, c: float, radius: float) -> Atoms:
    """Build a wurtzite supercell large enough to carve a sphere of given radius."""
    primitive = bulk("GaN", "wurtzite", a=a, c=c)
    rep_a = int(np.ceil(2 * radius / a)) + 2
    rep_c = int(np.ceil(2 * radius / c)) + 2
supercell = make_supercell(primitive, [[rep_a, 0, 0], [0, rep_a, 0], [0, 0, rep_c]])
supercell.center()
    return supercell


def substitute_for_alloy(atoms: Atoms, fraction_zno: float, rng: np.random.Generator) -> None:
    """Substitute Ga->Zn and N->O to achieve target ZnO fraction (in-place)."""
    if fraction_zno == 0.0:
        return
    symbols = atoms.get_chemical_symbols()
    ga_indices = [i for i, s in enumerate(symbols) if s == "Ga"]
    n_indices = [i for i, s in enumerate(symbols) if s == "N"]
    assert len(ga_indices) == len(n_indices), "Wurtzite must have equal cations/anions"
    n_substitute = int(round(len(ga_indices) * fraction_zno))
    if n_substitute == 0:
        return
    ga_to_zn = rng.choice(ga_indices, size=n_substitute, replace=False)
    n_to_o = rng.choice(n_indices, size=n_substitute, replace=False)
    for idx in ga_to_zn:
        atoms[idx].symbol = "Zn"
    for idx in n_to_o:
        atoms[idx].symbol = "O"


# =============================================================================
# PASSIVATION FUNCTIONS
# =============================================================================

def find_dangling_bonds(supercell: Atoms, valid_indices: np.ndarray) -> List[dict]:
    """Identify dangling bonds by comparing QD atoms to their bulk neighbors."""
    valid_set = set(valid_indices)
    nl_bulk = NeighborList(
        [BOND_CUTOFF / 2] * len(supercell),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    nl_bulk.update(supercell)
    passivants = []
    for idx in valid_indices:
        neighbor_indices, offsets = nl_bulk.get_neighbors(idx)
        for i, neighbor_idx in enumerate(neighbor_indices):
            if neighbor_idx not in valid_set:
                origin_pos = supercell.positions[idx]
                neighbor_pos = supercell.positions[neighbor_idx] + np.dot(offsets[i], supercell.get_cell())
                bond_vec = neighbor_pos - origin_pos
                bond_length = np.linalg.norm(bond_vec)
                unit_vec = bond_vec / bond_length
                h_pos = origin_pos + unit_vec * (bond_length * BOND_LENGTH_SCALING)
                parent_symbol = supercell[idx].symbol
                charge = PASSIVANT_CHARGES[parent_symbol]
                passivants.append({"position": h_pos, "charge": charge, "parent_symbol": parent_symbol})
    return passivants


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_quantum_dot(radius: float, fraction_zno: float, seed: int) -> Tuple[Atoms, dict]:
    """Generate a single passivated quantum dot structure."""
    rng = np.random.default_rng(seed)
    a, c = vegards_law(fraction_zno)
    supercell = build_bulk_supercell(a, c, radius)
    substitute_for_alloy(supercell, fraction_zno, rng)

    supercell_center = supercell.get_cell() @ np.array([0.5, 0.5, 0.5])
positions = supercell.get_positions()
    distances = np.linalg.norm(positions - supercell_center, axis=1)
mask = distances <= radius
    valid_indices = np.where(mask)[0]

    passivants = find_dangling_bonds(supercell, valid_indices)
    qd_uncentered = supercell[mask].copy()

    for p in passivants:
        h_atom = Atoms("H", positions=[p["position"]])
        h_atom.set_initial_charges([p["charge"]])
        qd_uncentered += h_atom

    qd_uncentered.center(vacuum=5.0)
    passivated_qd = qd_uncentered

    symbols = passivated_qd.get_chemical_symbols()
    n_ga = symbols.count("Ga")
    n_zn = symbols.count("Zn")
    n_n = symbols.count("N")
    n_o = symbols.count("O")
    n_h = symbols.count("H")

    metadata = {
        "radius_ang": radius,
        "fraction_zno": fraction_zno,
        "seed": seed,
        "lattice_a_ang": a,
        "lattice_c_ang": c,
        "n_core_atoms": n_ga + n_zn + n_n + n_o,
        "n_passivants": n_h,
        "n_total_atoms": len(passivated_qd),
        "n_ga": n_ga,
        "n_zn": n_zn,
        "n_n": n_n,
        "n_o": n_o,
    }
    return passivated_qd, metadata


def generate_dataset(radii: Sequence[float], fractions: Sequence[float], seeds: Sequence[int], output_dir: Path) -> None:
    """Generate a dataset of quantum dot structures with varying parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    structure_count = 0

    for radius in radii:
        for fraction in fractions:
            seeds_to_use = seeds if 0.0 < fraction < 1.0 else [seeds[0]]
            for seed in seeds_to_use:
                print(f"Generating: r={radius} A, x_ZnO={fraction:.2f}, seed={seed}")
                qd, metadata = generate_quantum_dot(radius, fraction, seed)

                if fraction == 0.0:
                    comp_str = "GaN"
                elif fraction == 1.0:
                    comp_str = "ZnO"
                else:
                    comp_str = f"GaN{1-fraction:.2f}_ZnO{fraction:.2f}"

                filename = f"QD_{comp_str}_r{radius:.0f}A_s{seed}.vasp"
                filepath = output_dir / filename
                write(filepath, qd, format="vasp", vasp5=True, direct=True, sort=True)

                metadata["filename"] = filename
                manifest_rows.append(metadata)
                structure_count += 1
                print(f"  -> {metadata['n_core_atoms']} core + {metadata['n_passivants']} H")

    manifest_path = output_dir / "qd_manifest.csv"
    assert len(manifest_rows) > 0, "No structures generated"

    fieldnames = list(manifest_rows[0].keys())
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nGenerated {structure_count} structures in {output_dir}")
    print(f"Manifest written to {manifest_path}")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def get_bond_key(sym1: str, sym2: str) -> Tuple[str, str]:
    """Return canonical bond key (alphabetically sorted)."""
    return tuple(sorted([sym1, sym2]))


def validate_structure(atoms: Atoms, verbose: bool = True) -> Dict[str, any]:
    """
    Validate a quantum dot structure for physical reasonableness.

    Checks:
    1. No overlapping atoms (minimum distance > 0.5 A)
    2. Bond lengths within expected ranges
    3. Stoichiometry (equal cations and anions in core)
    4. Passivant connectivity (H atoms bonded to exactly one core atom)
    5. Coordination numbers (core atoms have 1-4 neighbors)
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    n_atoms = len(atoms)

    results = {"valid": True, "errors": [], "warnings": [], "statistics": {}}

    cation_indices = [i for i, s in enumerate(symbols) if s in ("Ga", "Zn")]
    anion_indices = [i for i, s in enumerate(symbols) if s in ("N", "O")]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    core_indices = cation_indices + anion_indices

    results["statistics"]["n_cations"] = len(cation_indices)
    results["statistics"]["n_anions"] = len(anion_indices)
    results["statistics"]["n_passivants"] = len(h_indices)
    results["statistics"]["n_total"] = n_atoms

    if len(cation_indices) != len(anion_indices):
        results["warnings"].append(
            f"Stoichiometry mismatch: {len(cation_indices)} cations vs {len(anion_indices)} anions"
        )

    min_allowed_distance = 0.5
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_allowed_distance:
                results["errors"].append(
                    f"Overlapping atoms: {symbols[i]}({i}) and {symbols[j]}({j}) at {dist:.3f} A"
                )
                results["valid"] = False

    cutoff = 2.5
    nl = NeighborList([cutoff / 2] * n_atoms, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    bond_lengths = {key: [] for key in EXPECTED_BOND_LENGTHS}
    coordination_numbers = []

    for i in core_indices:
        neighbors, _ = nl.get_neighbors(i)
        coord = 0
        for n_idx in neighbors:
            n_sym = symbols[n_idx]
            dist = np.linalg.norm(positions[i] - positions[n_idx])
            if dist < cutoff:
                bond_key = get_bond_key(symbols[i], n_sym)
                if bond_key in bond_lengths:
                    bond_lengths[bond_key].append(dist)
                coord += 1
        coordination_numbers.append(coord)

    for bond_type, lengths in bond_lengths.items():
        if len(lengths) == 0:
            continue
        mean_len = np.mean(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        expected_min, expected_max = EXPECTED_BOND_LENGTHS[bond_type]

        results["statistics"][f"bond_{bond_type[0]}_{bond_type[1]}_mean"] = mean_len
        results["statistics"][f"bond_{bond_type[0]}_{bond_type[1]}_count"] = len(lengths)

        tolerance = 0.1 * (expected_max - expected_min + 0.5)
        if min_len < expected_min - tolerance or max_len > expected_max + tolerance:
            results["warnings"].append(
                f"{bond_type[0]}-{bond_type[1]} bonds outside expected range: "
                f"{min_len:.3f}-{max_len:.3f} A (expected {expected_min:.2f}-{expected_max:.2f} A)"
            )

    if coordination_numbers:
        results["statistics"]["coord_mean"] = np.mean(coordination_numbers)
        results["statistics"]["coord_min"] = np.min(coordination_numbers)
        results["statistics"]["coord_max"] = np.max(coordination_numbers)
        invalid_coord = [c for c in coordination_numbers if c < 1 or c > 4]
        if invalid_coord:
            results["warnings"].append(f"Found {len(invalid_coord)} atoms with unusual coordination (not 1-4)")

    isolated_h = 0
    multi_bonded_h = 0
    for h_idx in h_indices:
        neighbors, _ = nl.get_neighbors(h_idx)
        core_neighbors = [n for n in neighbors if n in core_indices]
        if len(core_neighbors) == 0:
            isolated_h += 1
        elif len(core_neighbors) > 1:
            multi_bonded_h += 1

    if isolated_h > 0:
        results["errors"].append(f"{isolated_h} H atoms not bonded to any core atom")
        results["valid"] = False
    if multi_bonded_h > 0:
        results["warnings"].append(f"{multi_bonded_h} H atoms bonded to multiple core atoms")

    if len(core_indices) > 0:
        results["statistics"]["passivant_to_core_ratio"] = len(h_indices) / len(core_indices)

    if verbose:
        print("\n" + "=" * 60)
        print("STRUCTURE VALIDATION REPORT")
        print("=" * 60)
        print(f"Total atoms: {n_atoms}")
        print(f"  Cations (Ga/Zn): {len(cation_indices)}")
        print(f"  Anions (N/O):    {len(anion_indices)}")
        print(f"  Passivants (H):  {len(h_indices)}")

        if "coord_mean" in results["statistics"]:
            print(f"\nCoordination: mean={results['statistics']['coord_mean']:.2f}, "
                  f"range=[{results['statistics']['coord_min']}, {results['statistics']['coord_max']}]")

        print("\nBond lengths:")
        for bond_type in EXPECTED_BOND_LENGTHS:
            key_mean = f"bond_{bond_type[0]}_{bond_type[1]}_mean"
            key_count = f"bond_{bond_type[0]}_{bond_type[1]}_count"
            if key_mean in results["statistics"]:
                print(f"  {bond_type[0]}-{bond_type[1]}: {results['statistics'][key_mean]:.3f} A "
                      f"(n={results['statistics'][key_count]})")

        if results["errors"]:
            print(f"\nERRORS ({len(results['errors'])}):")
            for err in results["errors"]:
                print(f"  [X] {err}")

        if results["warnings"]:
            print(f"\nWARNINGS ({len(results['warnings'])}):")
            for warn in results["warnings"]:
                print(f"  [!] {warn}")

        if results["valid"] and not results["warnings"]:
            print("\n[OK] Structure passed all validation checks.")
        elif results["valid"]:
            print("\n[OK] Structure valid with warnings.")
        else:
            print("\n[FAIL] Structure has critical errors.")
        print("=" * 60)

    return results


def validate_dataset(output_dir: Path, verbose: bool = True) -> None:
    """Validate all structures in a generated dataset."""
    manifest_path = output_dir / "qd_manifest.csv"
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nValidating {len(rows)} structures in {output_dir}\n")

    all_valid = True
    scaling_data = []

    for row in rows:
        filepath = output_dir / row["filename"]
        assert filepath.exists(), f"Structure file not found: {filepath}"

        if verbose:
            print(f"\n--- {row['filename']} ---")
            print(f"Radius: {row['radius_ang']} A, ZnO fraction: {row['fraction_zno']}")

        atoms = read(filepath)
        results = validate_structure(atoms, verbose=verbose)

        if not results["valid"]:
            all_valid = False

        scaling_data.append({
            "radius": float(row["radius_ang"]),
            "n_core": results["statistics"]["n_cations"] + results["statistics"]["n_anions"],
            "n_passivants": results["statistics"]["n_passivants"],
        })

    print("\n" + "=" * 60)
    print("SCALING LAW ANALYSIS")
    print("=" * 60)
    print("Expected: n_core ~ r^3, n_passivants ~ r^2")
    print("\nRadius (A)  |  n_core  |  n_pass  |  ratio")
    print("-" * 45)

    for d in sorted(scaling_data, key=lambda x: x["radius"]):
        ratio = d["n_passivants"] / d["n_core"] if d["n_core"] > 0 else 0
        print(f"  {d['radius']:5.1f}     |   {d['n_core']:5d}  |   {d['n_passivants']:4d}   |  {ratio:.3f}")

    radii = np.array([d["radius"] for d in scaling_data])
    n_cores = np.array([d["n_core"] for d in scaling_data])
    n_pass = np.array([d["n_passivants"] for d in scaling_data])

    if len(set(radii)) > 1:
        log_r = np.log(radii)
        log_core = np.log(n_cores)
        log_pass = np.log(n_pass)
        core_exponent = np.polyfit(log_r, log_core, 1)[0]
        pass_exponent = np.polyfit(log_r, log_pass, 1)[0]
        print(f"\nFitted scaling exponents:")
        print(f"  n_core ~ r^{core_exponent:.2f}  (expected: 3.0)")
        print(f"  n_pass ~ r^{pass_exponent:.2f}  (expected: 2.0)")
        if abs(core_exponent - 3.0) > 0.5:
            print("  [!] WARNING: Core scaling deviates significantly from r^3")
        if abs(pass_exponent - 2.0) > 0.5:
            print("  [!] WARNING: Passivant scaling deviates significantly from r^2")

    print("=" * 60)

    if all_valid:
        print("\n[OK] All structures passed validation.")
            else:
        print("\n[FAIL] Some structures have errors. Review above.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate passivated wurtzite quantum dot dataset for VASP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--radii", type=float, nargs="+", default=[8.0, 10.0, 12.0],
                        help="QD radii in Angstroms (default: 8 10 12)")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="ZnO fractions for alloy (default: 0.0 0.5 1.0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43],
                        help="Random seeds for alloy sampling (default: 42 43)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing structures instead of generating new ones.")
    parser.add_argument("--validate-file", type=Path, default=None,
                        help="Validate a single POSCAR/VASP file.")
    return parser.parse_args()


def main() -> None:
    """Entry point for quantum dot generation and validation."""
    args = parse_args()

    if args.validate_file is not None:
        assert args.validate_file.exists(), f"File not found: {args.validate_file}"
        atoms = read(args.validate_file)
        validate_structure(atoms, verbose=True)
        return

    if args.validate:
        validate_dataset(args.output_dir.resolve(), verbose=True)
        return

    assert all(r > 0 for r in args.radii), "Radii must be positive"
    assert all(0.0 <= f <= 1.0 for f in args.fractions), "Fractions must be in [0, 1]"

    generate_dataset(
        radii=args.radii,
        fractions=args.fractions,
        seeds=args.seeds,
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
