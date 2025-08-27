#!/usr/bin/env python3
"""
band gap calculations using GPAW
https://gpaw.readthedocs.io/tutorialsexercises/electronic/bandstructures/bandstructures.html#bandstructures
https://gpaw.readthedocs.io/documentation/parallel_runs/parallel_runs.html 

"""
import argparse
import logging
import numpy as np
import pandas as pd
import ase.io
from gpaw import GPAW, PW, FermiDirac
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def calculate_bandgap(atoms, method='PBE'):
    """Calculate band gap using GPAW"""
    # Set up calculator
    calc = GPAW(
        mode=PW(200),
        xc=method,
        kpts=(2, 2, 2), 
        occupations=FermiDirac(0.01),
        txt=None  # Suppress output
    )
    
    atoms.calc = calc
    atoms.get_potential_energy()
    
    # Get eigenvalues at gamma point
    eigenvalues = calc.get_eigenvalues(kpt=0)
    nelec = calc.get_number_of_electrons()
    bs = calc.band_structure()
    bs.plot(filename='bandstructure.png', show=True, emax=10.0)
    # Find HOMO and LUMO
    homo_idx = int(nelec // 2) - 1
    lumo_idx = int(nelec // 2)
    
    if homo_idx >= 0 and lumo_idx < len(eigenvalues):
        homo_energy = eigenvalues[homo_idx]
        lumo_energy = eigenvalues[lumo_idx]
        band_gap = lumo_energy - homo_energy
        return band_gap, homo_energy, lumo_energy, bs
    else:
        return float('nan'), float('nan'), float('nan')

def main():
    parser = argparse.ArgumentParser(description='Calculate band gaps using GPAW')
    parser.add_argument('--xyz_file', default='trainall.xyz')
    parser.add_argument('--output', default='bandgap_results.csv')
    parser.add_argument('--method', default='PBE', choices=['PBE', 'PBE0', 'HSE06'])
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    # Read structures
    if args.test:
        structures = list(ase.io.read(args.xyz_file, index=':3'))
        logger.info(f"Testing with {len(structures)} structures")
    else:
        structures = list(ase.io.read(args.xyz_file, index=':'))
        logger.info(f"Processing {len(structures)} structures")
    
    results = []
    
    for i, atoms in enumerate(structures):
        logger.info(f"Structure {i+1}: {atoms.get_chemical_formula()}")
        
        try:
            band_gap, homo, lumo, bs = calculate_bandgap(atoms, args.method)
            logger.info(f"  Band gap: {band_gap:.3f} eV")
            
            results.append({
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'band_gap': band_gap,
                'homo_energy': homo,
                'lumo_energy': lumo,
                'bs': 'calculated' if bs is not None else 'None'
            })
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'band_gap': float('nan'),
                'homo_energy': float('nan'),
                'lumo_energy': float('nan')
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    valid_gaps = df['band_gap'].dropna()
    if len(valid_gaps) > 0:
        logger.info(f"Mean band gap: {valid_gaps.mean():.3f} ± {valid_gaps.std():.3f} eV")
    else:
        logger.info("No valid band gaps calculated")

if __name__ == "__main__":
    main()
