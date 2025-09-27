"""
Evaluate a parameter TOML against bulk/supercell test structures.

Usage:
  pixi run python -m src.scripts.eval_bulk --system big --params config/gfn1-base.toml \
      --xyz trainall.xyz --results test_structs/results.csv --out results/eval.csv

This script loads the XYZ, attaches reference targets from results.csv if present,
runs TBLite (GFN1) with the provided param file to compute energies/bandgaps, and
reports RMSEs and per-structure outputs.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ase.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calculators.tblite_ase_calculator import TBLiteASECalculator


def attach_references(structures, results_csv: Path) -> None:
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    n = min(len(df), len(structures))
    for i in range(n):
        row = df.iloc[i]
        atoms = structures[i]
        if not hasattr(atoms, 'info'):
            continue
        if 'FreeEnergy(eV)' in row:
            try:
                atoms.info['ref_energy_eV'] = float(row['FreeEnergy(eV)'])
            except Exception:
                atoms.info['ref_energy_eV'] = np.nan
        if 'Bandgap(eV)' in row:
            try:
                atoms.info['ref_gap_eV'] = float(row['Bandgap(eV)'])
            except Exception:
                atoms.info['ref_gap_eV'] = np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default='big')
    ap.add_argument('--params', required=True, help='Path to TOML parameter file')
    ap.add_argument('--xyz', default='trainall.xyz')
    ap.add_argument('--results', default='test_structs/results.csv')
    ap.add_argument('--out', default='results/eval.csv')
    args = ap.parse_args()

    xyz_path = Path(args.xyz)
    res_path = Path(args.results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    structures = list(ase.io.read(str(xyz_path), index=':'))
    attach_references(structures, res_path)

    energies_ev = []
    gaps_ev = []
    ref_e = []
    ref_g = []

    for i, atoms in enumerate(structures):
        try:
            calc = TBLiteASECalculator(
                param_file=args.params,
                method='gfn1',
                electronic_temperature=400.0,
                charge=0.0,
                spin=0
            )
            e_h = calc.get_potential_energy(atoms)
            e_ev = float(e_h) * 27.211386245988
            energies_ev.append(e_ev)
            gap = float(calc.results.get('bandgap', np.nan))
            gaps_ev.append(gap)
        except Exception as exc:
            energies_ev.append(np.nan)
            gaps_ev.append(np.nan)
        # references
        info = atoms.info if hasattr(atoms, 'info') else {}
        ref_e.append(float(info.get('ref_energy_eV', np.nan)))
        ref_g.append(float(info.get('ref_gap_eV', np.nan)))

    df = pd.DataFrame({
        'calc_energy_eV': energies_ev,
        'ref_energy_eV': ref_e,
        'calc_gap_eV': gaps_ev,
        'ref_gap_eV': ref_g,
    })

    # RMSE metrics where both sides present
    def rmse(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            return np.nan
        return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))

    e_rmse = rmse(df['calc_energy_eV'].values - np.nanmin(df['calc_energy_eV'].values),
                  df['ref_energy_eV'].values - np.nanmin(df['ref_energy_eV'].values))
    g_rmse = rmse(df['calc_gap_eV'].values, df['ref_gap_eV'].values)

    df.to_csv(out_path, index=False)

    print(f"Saved per-structure results to {out_path}")
    print(f"Energy RMSE (relative): {e_rmse:.6f} eV")
    if np.isfinite(g_rmse):
        print(f"Bandgap RMSE: {g_rmse:.6f} eV")
    else:
        print("Bandgap RMSE: N/A")


if __name__ == '__main__':
    main()


