"""
Dataset generator for wurtzite GaN/ZnO mixtures.

The script:
- Reads `GaN.poscar` and `ZnO.poscar` from the project `data` directory.
- Validates lattice parameters against literature values (GaN: a=3.189 Å,
  c=5.185 Å from Y. T. Tsang et al., J. Appl. Phys. 79, 1996; ZnO: a=3.249 Å,
  c=5.206 Å from Ü. Özgür et al., J. Appl. Phys. 98, 041301 (2005)).
- Builds wurtzite supercells and substitutes Ga↔Zn and N↔O to reach requested
  compositions.
- Writes POSCAR files plus a CSV manifest describing each generated structure.

Usage example:
  pixi run python src/scripts/gan_zno.py --fractions 0 0.2 0.4 0.5 0.6 0.8 1.0 \
    --supercell 3 3 2 --output-dir data/gan_zno_supercells
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.io import read, write

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "gan_zno_supercells"
DEFAULT_FRACTIONS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SUPERCELL = (3, 3, 2)
LATTICE_TOLERANCE_ANG = 0.05

GAN_LATTICE = {
    "a": 3.189,
    "c": 5.185,
    "source": "Y. T. Tsang et al., J. Appl. Phys. 79 (1996)",
}
ZNO_LATTICE = {
    "a": 3.249,
    "c": 5.206,
    "source": "Ü. Özgür et al., J. Appl. Phys. 98, 041301 (2005)",
}


def read_poscar(path: Path) -> Atoms:
    """Load a POSCAR file and ensure it contains atoms."""
    atoms = read(path)
    assert len(atoms) > 0, f"File {path} did not contain any atoms."
    return atoms


def validate_lattice(atoms: Atoms, expected: dict, label: str) -> Tuple[float, float]:
    """Assert that lattice constants agree with literature within tolerance."""
    cell_params = atoms.cell.cellpar()
    measured_a = float(cell_params[0])
    measured_c = float(cell_params[2])
    assert (
        abs(measured_a - expected["a"]) <= LATTICE_TOLERANCE_ANG
    ), f"{label} a={measured_a:.3f} Å deviates from {expected['a']:.3f} Å ({expected['source']})."
    assert (
        abs(measured_c - expected["c"]) <= LATTICE_TOLERANCE_ANG
    ), f"{label} c={measured_c:.3f} Å deviates from {expected['c']:.3f} Å ({expected['source']})."
    return measured_a, measured_c


def vegards_target(fraction: float) -> Tuple[float, float]:
    """Interpolate lattice constants via Vegard's law for a given ZnO fraction."""
    assert 0.0 <= fraction <= 1.0, "Fractions must be within [0, 1]."
    target_a = GAN_LATTICE["a"] + (ZNO_LATTICE["a"] - GAN_LATTICE["a"]) * fraction
    target_c = GAN_LATTICE["c"] + (ZNO_LATTICE["c"] - GAN_LATTICE["c"]) * fraction
    return target_a, target_c


def build_supercell(atoms: Atoms, repeats: Sequence[int]) -> Atoms:
    """Expand a primitive cell into a supercell."""
    assert len(repeats) == 3, "Supercell repeat must have three integers."
    assert all(rep > 0 for rep in repeats), "Supercell repeats must be positive."
    return atoms.repeat(tuple(int(rep) for rep in repeats))


def substitute_species(
    atoms: Atoms, indices: Sequence[int], new_symbol: str, count: int, rng: np.random.Generator
) -> None:
    """Replace `count` atoms selected from indices with `new_symbol`."""
    assert count <= len(indices), "Substitution count exceeds available sites."
    if count == 0:
        return
    chosen = rng.choice(indices, size=count, replace=False)
    for idx in chosen:
        atoms[int(idx)].symbol = new_symbol


def mix_gan_zno(supercell: Atoms, fraction: float, rng: np.random.Generator) -> Atoms:
    """Substitute Ga→Zn and N→O to achieve the requested ZnO fraction."""
    symbols = supercell.get_chemical_symbols()
    ga_sites = [i for i, sym in enumerate(symbols) if sym == "Ga"]
    n_sites = [i for i, sym in enumerate(symbols) if sym == "N"]
    assert len(ga_sites) == len(n_sites), "Wurtzite cell should have equal cations/anions."

    target_zn = int(round(len(ga_sites) * fraction))
    target_o = int(round(len(n_sites) * fraction))

    mixed = supercell.copy()
    substitute_species(mixed, ga_sites, "Zn", target_zn, rng)
    substitute_species(mixed, n_sites, "O", target_o, rng)
    return mixed


def rescale_cell_for_fraction(atoms: Atoms, repeats: Sequence[int], fraction: float) -> Tuple[float, float]:
    """Apply Vegard's law scaling to match in-plane (a) and out-of-plane (c) targets."""
    target_a, target_c = vegards_target(fraction)
    cell = atoms.get_cell()
    current_lengths = cell.lengths()

    scale_x = (target_a * repeats[0]) / current_lengths[0]
    scale_y = (target_a * repeats[1]) / current_lengths[1]
    scale_z = (target_c * repeats[2]) / current_lengths[2]

    scaled_cell = cell.copy()
    scaled_cell[0] *= scale_x
    scaled_cell[1] *= scale_y
    scaled_cell[2] *= scale_z
    atoms.set_cell(scaled_cell, scale_atoms=True)
    return target_a, target_c


def write_manifest(manifest_path: Path, rows: Iterable[dict]) -> None:
    """Write a CSV manifest describing generated structures."""
    rows = list(rows)
    assert len(rows) > 0, "No dataset entries to write."
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    fractions: Sequence[float],
    supercell_repeats: Sequence[int],
    output_dir: Path,
    seed: int,
) -> None:
    """Create POSCAR files for GaN/ZnO mixtures and write a manifest."""
    assert output_dir.is_absolute(), "Output directory must be an absolute path."
    output_dir.mkdir(parents=True, exist_ok=True)

    gan_path = DATA_DIR / "GaN.poscar"
    zno_path = DATA_DIR / "ZnO.poscar"
    assert gan_path.exists(), f"Missing GaN POSCAR at {gan_path}."
    assert zno_path.exists(), f"Missing ZnO POSCAR at {zno_path}."

    gan_atoms = read_poscar(gan_path)
    zno_atoms = read_poscar(zno_path)

    validate_lattice(gan_atoms, GAN_LATTICE, "GaN")
    validate_lattice(zno_atoms, ZNO_LATTICE, "ZnO")

    base_supercell = build_supercell(gan_atoms, supercell_repeats)
    rng = np.random.default_rng(seed)

    manifest_rows = []
    for fraction in fractions:
        mixed = mix_gan_zno(base_supercell, fraction, rng)
        target_a, target_c = rescale_cell_for_fraction(mixed, supercell_repeats, fraction)

        counts = Counter(mixed.get_chemical_symbols())
        fname = f"gan_zno_x{fraction:.2f}_sc{'x'.join(str(r) for r in supercell_repeats)}.vasp"
        output_path = output_dir / fname
        write(output_path, mixed, format="vasp", vasp5=True, direct=True, sort=True)

        manifest_rows.append(
            {
                "fraction_zno": f"{fraction:.3f}",
                "ga": counts.get("Ga", 0),
                "zn": counts.get("Zn", 0),
                "n": counts.get("N", 0),
                "o": counts.get("O", 0),
                "supercell": "x".join(str(rep) for rep in supercell_repeats),
                "target_a_ang": f"{target_a:.4f}",
                "target_c_ang": f"{target_c:.4f}",
                "poscar": output_path.name,
            }
        )

    write_manifest(output_dir / "gan_zno_manifest.csv", manifest_rows)
    print(f"Generated {len(manifest_rows)} structures in {output_dir}")


def parse_args() -> argparse.Namespace:
    """Argument parser for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate GaN/ZnO wurtzite supercell dataset using ASE."
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="ZnO fraction values (0–1) to generate.",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=list(DEFAULT_SUPERCELL),
        metavar=("NX", "NY", "NZ"),
        help="Supercell repeats for a single wurtzite cell.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write POSCAR files and manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for deterministic atom substitutions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = tuple(float(val) for val in args.fractions)
    assert all(0.0 <= frac <= 1.0 for frac in fractions), "All fractions must be in [0, 1]."
    supercell_repeats = tuple(int(rep) for rep in args.supercell)
    generate_dataset(fractions, supercell_repeats, args.output_dir.resolve(), args.seed)


if __name__ == "__main__":
    main()
