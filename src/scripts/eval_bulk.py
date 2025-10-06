"""
Evaluate a parameter TOML against bulk/supercell test structures.

Usage:
pixi run python src/scripts/eval_bulk.py --system big --params config/gfn1-base.toml --xyz val_lind50_eq.xyz --results test_structs/results.csv --out results/base.csv --skip-energy

pixi run python src/scripts/eval_bulk.py --system big --params results/gad/big_gad.toml --xyz val_lind50_eq.xyz --results test_structs/results.csv --out results/gad.csv 

pixi run python src/scripts/eval_bulk.py  --params results/pso/big_pso.toml --out results/pso2.csv 

pixi run python src/scripts/eval_bulk.py  --params results/ga/big_ga.toml --out results/ga.csv 

pixi run python src/scripts/eval_bulk.py  --params results/bayes/big_bayes.toml --out results/bayes.csv 

"""
import argparse
import sys
from pathlib import Path
import ase.io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.dft.bandgap import bandgap as ase_bandgap
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calculators.tblite_ase_calculator import TBLiteASECalculator

def attach_references(structures, results_csv: Path) -> None:
    if not results_csv.exists():
        print(f"Reference CSV not found: {results_csv}")
        return
    df = pd.read_csv(results_csv)
    # convert to floates
    if 'FreeEnergy(eV)' in df.columns:
        df['FreeEnergy(eV)'] = pd.to_numeric(df['FreeEnergy(eV)'], errors='coerce')
    if 'Bandgap(eV)' in df.columns:
        df['Bandgap(eV)'] = pd.to_numeric(df['Bandgap(eV)'], errors='coerce')
    n = min(len(df), len(structures))
    ref_e_count = 0
    ref_g_count = 0
    for i in range(n):
        atoms = structures[i]
        if not hasattr(atoms, 'info'):
            continue
        # Free energy
        if 'FreeEnergy(eV)' in df.columns:
            val_e = df.at[i, 'FreeEnergy(eV)']
            if pd.notna(val_e):
                try:
                    atoms.info['ref_energy_eV'] = float(val_e)
                    ref_e_count += 1
                except Exception:
                    atoms.info['ref_energy_eV'] = np.nan
        # Bandgap
        if 'Bandgap(eV)' in df.columns:
            val_g = df.at[i, 'Bandgap(eV)']
            if pd.notna(val_g):
                try:
                    atoms.info['ref_gap_eV'] = float(val_g)
                    ref_g_count += 1
                except Exception:
                    atoms.info['ref_gap_eV'] = np.nan
    print(f"Attached references from {results_csv}: {n} structures; ref_energy count={ref_e_count}, ref_gap count={ref_g_count}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default='big')
    ap.add_argument('--params', required=True, help='Path to TOML parameter file')
    ap.add_argument('--xyz', default='val_lind50_eq.xyz')
    ap.add_argument('--results', default='test_structs/results.csv')
    ap.add_argument('--out', default='results/eval.csv')
    ap.add_argument('--skip-energy', action='store_true', help='Skip computing free energy and related plots; reuse from existing out if available')
    args = ap.parse_args()

    xyz_path = Path(args.xyz)
    res_path = Path(args.results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    structures = list(ase.io.read(str(xyz_path), index=':'))
    attach_references(structures, res_path)

    # reuse logic
    prev_df = None
    reuse_energy = False
    prev_energies = None
    if out_path.exists():
        try:
            prev_df = pd.read_csv(out_path)
            if len(prev_df) == len(structures) and 'calc_energy_eV' in prev_df.columns:
                if np.isfinite(prev_df['calc_energy_eV'].values).any():
                    prev_energies = prev_df['calc_energy_eV'].values.astype(float)
                    reuse_energy = True
        except Exception:
            reuse_energy = False

    energies_ev = [] if not reuse_energy else prev_energies.tolist()
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
            # Trigger calculation; we always run TBLite to get bandgap
            e_h = calc.get_potential_energy(atoms)
            if not reuse_energy and not args.skip_energy:
                e_ev = float(e_h) * 27.211386245988
                print("energy append success")
                energies_ev.append(e_ev)
            # Prefer calculator result first
            try:
                gap = float(calc.results.get('bandgap', np.nan))
                print("bandgap success")
            except Exception:
                print("hmmm")
                gap = np.nan
            gaps_ev.append(gap)
        except Exception as exc:
            print('oops! something went wrong in the calculation loop')
            if not reuse_energy and not args.skip_energy:
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

    # RMSE 
    def rmse(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            return np.nan
        return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))

    e_rmse = rmse(df['calc_energy_eV'].values - np.nanmin(df['calc_energy_eV'].values),
                  df['ref_energy_eV'].values - np.nanmin(df['ref_energy_eV'].values)) if not args.skip_energy else np.nan
    g_rmse = rmse(df['calc_gap_eV'].values, df['ref_gap_eV'].values)

    df.to_csv(out_path, index=False)

    print(f"Saved per-structure results to {out_path}")
    if np.isfinite(e_rmse):
        print(f"Energy RMSE (relative): {e_rmse:.6f} eV")
    else:
        print("Energy RMSE (relative): skipped")
    if np.isfinite(g_rmse):
        print(f"Bandgap RMSE: {g_rmse:.6f} eV")
    else:
        print("Bandgap RMSE: N/A")

    # Diagnostics for bandgaps
    n_total = len(df)
    n_calc_g = int(np.isfinite(df['calc_gap_eV'].values).sum())
    n_ref_g = int(np.isfinite(df['ref_gap_eV'].values).sum())
    print(f"Bandgap counts — calc: {n_calc_g}/{n_total}, ref: {n_ref_g}/{n_total}")

    # Save scatter plot: reference vs calculated bandgap
    calc_gap = df['calc_gap_eV'].values
    ref_gap = df['ref_gap_eV'].values
    finite_mask = np.isfinite(calc_gap) & np.isfinite(ref_gap)
    if np.any(finite_mask):
        x = ref_gap[finite_mask]
        y = calc_gap[finite_mask]
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=12, alpha=0.7)
        # Diagonal reference line
        vmin = float(min(np.min(x), np.min(y)))
        vmax = float(max(np.max(x), np.max(y)))
        span = vmax - vmin
        pad = 0.05 * (span if span > 0 else 1.0)
        lo = vmin - pad
        hi = vmax + pad
        plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel('Reference Bandgap (eV)')
        plt.ylabel('Calculated Bandgap (eV)')
        title_rmse = f" (RMSE={g_rmse:.3f} eV)" if np.isfinite(g_rmse) else ""
        plt.title(f'Bandgap: Predicted vs Actual{title_rmse}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_path.with_suffix('.bandgap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved bandgap plot to {plot_path}")
    else:
        print("Bandgap plot skipped: no finite values to plot")

    # Save scatter plot: reference vs calculated free energy (relative)
    energy_plot_path = out_path.with_suffix('.free_energy.png')
    skip_energy_plot = args.skip_energy or (reuse_energy and energy_plot_path.exists())
    if skip_energy_plot:
        print("Free energy plot skipped: reuse/skip enabled")
    else:
        calc_e = df['calc_energy_eV'].values
        ref_e_vals = df['ref_energy_eV'].values
        # Use relative energies, consistent with RMSE definition
        if np.any(np.isfinite(calc_e)) and np.any(np.isfinite(ref_e_vals)):
            # Compute per-array minima ignoring NaNs
            calc_min = np.nanmin(calc_e)
            ref_min = np.nanmin(ref_e_vals)
            calc_rel = calc_e - calc_min
            ref_rel = ref_e_vals - ref_min
            finite_mask_e = np.isfinite(calc_rel) & np.isfinite(ref_rel)
            if np.any(finite_mask_e):
                x = ref_rel[finite_mask_e]
                y = calc_rel[finite_mask_e]
                plt.figure(figsize=(6, 6))
                plt.scatter(x, y, s=12, alpha=0.7)
                # Diagonal reference line
                vmin = float(min(np.min(x), np.min(y)))
                vmax = float(max(np.max(x), np.max(y)))
                span = vmax - vmin
                pad = 0.05 * (span if span > 0 else 1.0)
                lo = vmin - pad
                hi = vmax + pad
                plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
                plt.xlim(lo, hi)
                plt.ylim(lo, hi)
                plt.xlabel('Reference Relative Free Energy (eV)')
                plt.ylabel('Calculated Relative Free Energy (eV)')
                title_rmse_e = f" (RMSE={e_rmse:.3f} eV)" if np.isfinite(e_rmse) else ""
                plt.title(f'Free Energy: Predicted vs Actual (relative){title_rmse_e}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved free energy plot to {energy_plot_path}")
            else:
                print("Energy plot skipped: no finite relative values to plot")
        else:
            print("Energy plot skipped: insufficient finite values")


if __name__ == '__main__':
    main()


