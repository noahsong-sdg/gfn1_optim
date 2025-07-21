#!/usr/bin/env python3
"""
Streamlined band gap calculation using PySCF.
Uses proper band structure API for gamma point calculations.
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import ase.io
from ase import Atoms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PySCF imports
import pyscf
import pyscf.pbc.gto
import pyscf.pbc.dft
from pyscf.pbc import dft as pbc_dft


def atoms_to_pyscf_cell(atoms: Atoms, basis: str = 'sto-3g') -> pyscf.pbc.gto.Cell:
    """Convert ASE atoms to PySCF cell for periodic calculations"""
    cell_params = atoms.get_cell()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    cell = pyscf.pbc.gto.Cell()
    cell.a = cell_params
    cell.unit = 'Angstrom'
    cell.basis = basis
    cell.verbose = 0
    
    # Add atoms
    for symbol, pos in zip(symbols, positions):
        cell.atom.append([symbol, pos.tolist()])
    
    cell.build()
    return cell


def calculate_bandgap(atoms: Atoms, method: str = 'pbe', basis: str = 'sto-3g') -> Dict[str, float]:
    """
    Calculate band gap using PySCF band structure API, following the canonical examplefromdocs.py pattern.
    Runs SCF, uses get_bands, and extracts VBM/CBM across all k-points.
    Returns both direct and indirect band gaps if possible.
    """
    try:
        cell = atoms_to_pyscf_cell(atoms, basis)
        if not cell.nbas:
            raise ValueError("Cell has no basis functions - check structure and basis set")
        # Use only gamma point for minimal case, but allow for extension
        kpts = np.array([[0, 0, 0]])
        
        mf = pbc_dft.RKS(cell, kpts)
        if method.lower() == 'pbe':
            mf.xc = 'pbe'
        elif method.lower() == 'pbe0':
            mf.xc = 'pbe0'
            mf.exxdiv = 'ewald'
        elif method.lower() == 'b3lyp':
            mf.xc = 'b3lyp'
            mf.exxdiv = 'ewald'
        else:
            mf.xc = method
        mf = mf.density_fit()
        mf.max_memory = 4000
        mf.conv_tol = 1e-5
        mf.max_cycle = 100
        mf.kernel()

        # For a full band structure, you would use a k-point path. Here, just gamma:
        band_kpts = kpts
        e_kn, kpts_bands = mf.get_bands(band_kpts)
        # e_kn shape: (n_kpts, n_bands)
        n_kpts, n_bands = e_kn.shape
        nelec = cell.nelectron
        nocc = nelec // 2  # spin-restricted

        vbm_list = []
        cbm_list = []
        direct_gaps = []
        for en in e_kn:
            occ = en[:nocc]
            unocc = en[nocc:]
            if len(occ) > 0:
                vbm = np.max(occ)
                vbm_list.append(vbm)
            if len(unocc) > 0:
                cbm = np.min(unocc)
                cbm_list.append(cbm)
            if len(occ) > 0 and len(unocc) > 0:
                direct_gaps.append(np.min(unocc) - np.max(occ))

        if vbm_list and cbm_list:
            global_vbm = np.max(vbm_list)
            global_cbm = np.min(cbm_list)
            indirect_gap = global_cbm - global_vbm
            direct_gap = np.min(direct_gaps) if direct_gaps else indirect_gap
            return {
                'indirect_gap': indirect_gap,
                'direct_gap': direct_gap,
                'global_vbm': global_vbm,
                'global_cbm': global_cbm,
                'method': method,
                'basis': basis,
                'converged': mf.converged,
                'total_energy': mf.e_tot,
                'error': None
            }
        return {
            'indirect_gap': 0.0,
            'direct_gap': 0.0,
            'global_vbm': 0.0,
            'global_cbm': 0.0,
            'method': method,
            'basis': basis,
            'converged': mf.converged,
            'total_energy': mf.e_tot,
            'error': 'Could not determine band gap'
        }
    except Exception as e:
        return {
            'indirect_gap': 0.0,
            'direct_gap': 0.0,
            'global_vbm': 0.0,
            'global_cbm': 0.0,
            'method': method,
            'basis': basis,
            'converged': False,
            'total_energy': 0.0,
            'error': str(e)
        }


def process_structures(xyz_file: str, output_file: str = None, method: str = 'pbe', 
                     basis: str = 'sto-3g') -> pd.DataFrame:
    """
    Process all structures in an XYZ file and compute band gaps
    
    Args:
        xyz_file: Path to XYZ file
        output_file: Path to output CSV file
        method: DFT functional to use
        basis: Basis set
        
    Returns:
        DataFrame with results
    """
    print(f"Reading structures from {xyz_file}...")
    structures = list(ase.io.read(xyz_file, index=':'))
    print(f"Found {len(structures)} structures")
    
    results = []
    start_time = time.time()
    
    for i, atoms in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}: {atoms.get_chemical_formula()}")
        
        # Get structure info
        structure_info = {
            'structure_id': i,
            'formula': atoms.get_chemical_formula(),
            'n_atoms': len(atoms),
            'volume': atoms.get_volume(),
            'cell_a': atoms.cell[0, 0] if atoms.cell.any() else 0.0,
            'cell_b': atoms.cell[1, 1] if atoms.cell.any() else 0.0,
            'cell_c': atoms.cell[2, 2] if atoms.cell.any() else 0.0,
        }
        
        # Calculate band gap
        calc_start_time = time.time()
        try:
            result = calculate_bandgap(atoms, method, basis)
            calc_time = time.time() - calc_start_time
            
        except Exception as e:
            calc_time = time.time() - calc_start_time
            logger.error(f"Calculation failed for structure {i+1}: {e}")
            result = {
                'indirect_gap': 0.0,
                'direct_gap': 0.0,
                'global_vbm': 0.0,
                'global_cbm': 0.0,
                'method': method,
                'basis': basis,
                'converged': False,
                'total_energy': 0.0,
                'error': str(e)
            }
        
        # Combine results
        combined_result = {**structure_info, **result, 'calculation_time': calc_time}
        results.append(combined_result)
        
        # Print progress
        if result['converged'] and result['error'] is None:
            print(f"  Band gap: {result['direct_gap']:.4f} eV (time: {calc_time/60:.1f} min)")
        else:
            print(f"  Failed to converge (time: {calc_time/60:.1f} min)")
            if result['error']:
                print(f"  Error: {result['error']}")
        
        # Save intermediate results every 10 structures
        if (i + 1) % 10 == 0:
            df_temp = pd.DataFrame(results)
            temp_file = f"temp_results_{i+1}.csv"
            df_temp.to_csv(temp_file, index=False)
            print(f"  Saved intermediate results to {temp_file}")
    
    # Create final DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if output_file is None:
        output_file = f"band_gap_results_{Path(xyz_file).stem}_{method}_{basis}.csv"
    
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    successful = df[df['converged'] == True]
    failed = df[df['converged'] == False]
    
    print(f"\n=== SUMMARY ===")
    print(f"Total structures: {len(structures)}")
    print(f"Successful calculations: {len(successful)}")
    print(f"Failed calculations: {len(failed)}")
    print(f"Success rate: {len(successful)/len(structures)*100:.1f}%")
    
    if len(successful) > 0:
        band_gaps = successful['direct_gap'].values
        print(f"Band gap range: {np.min(band_gaps):.3f} - {np.max(band_gaps):.3f} eV")
        print(f"Mean band gap: {np.mean(band_gaps):.3f} ± {np.std(band_gaps):.3f} eV")
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average time per structure: {total_time/len(structures)/60:.1f} minutes")
    
    return df


def main():
    """Main function to run band gap calculations"""
    parser = argparse.ArgumentParser(description='Calculate band gaps using PySCF')
    parser.add_argument('--xyz_file', default='trainall.xyz', 
                       help='Input XYZ file (default: trainall.xyz)')
    parser.add_argument('--output', default=None,
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--method', default='pbe',
                       choices=['pbe', 'pbe0', 'b3lyp'],
                       help='DFT functional (default: pbe)')
    parser.add_argument('--basis', default='sto-3g',
                       help='Basis set (default: sto-3g)')
    parser.add_argument('--test', action='store_true',
                       help='Test with first 3 structures only')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with first 3 structures...")
        structures = list(ase.io.read(args.xyz_file, index=':3'))
        test_file = 'test_structures.xyz'
        ase.io.write(test_file, structures)
        
        df = process_structures(test_file, args.output, args.method, args.basis)
        os.remove(test_file)
    else:
        df = process_structures(args.xyz_file, args.output, args.method, args.basis)
    
    return df


if __name__ == "__main__":
    main() 
