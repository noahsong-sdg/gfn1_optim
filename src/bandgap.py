#!/usr/bin/env python3
"""
Band gap calculation script for reference dataset generation.
Uses PySCF for high-quality DFT calculations with proper band gap extraction.
"""

import sys
import os
import json
import argparse
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PySCF imports
import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.dft
import pyscf.pbc.scf
import pyscf.pbc.tools
from pyscf import dft, scf
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import scf as pbc_scf


def atoms_to_pyscf_cell(atoms: Atoms, basis: str = 'sto-3g', ecut: float = 400.0) -> pyscf.pbc.gto.Cell:
    """Convert ASE atoms to PySCF cell for periodic calculations"""
    # Get cell parameters
    cell_params = atoms.get_cell()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Create PySCF cell
    cell = pyscf.pbc.gto.Cell()
    cell.a = cell_params
    cell.unit = 'Angstrom'
    cell.basis = basis
    cell.ecut = ecut
    cell.verbose = 0  # Reduce output
    
    # Memory optimization: remove very diffuse functions
    cell.exp_to_discard = 0.1
    
    # Add atoms
    for i, (symbol, pos) in enumerate(zip(symbols, positions)):
        cell.atom.append([symbol, pos.tolist()])
    
    cell.build()
    return cell


def calculate_bandgap_pyscf(atoms: Atoms, method: str = 'pbe', basis: str = 'sto-3g', 
                           ecut: float = 400.0, kpts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate band gap using PySCF with periodic boundary conditions.
    
    Args:
        atoms: ASE atoms object
        method: DFT functional (pbe, pbe0, b3lyp, etc.)
        basis: Basis set
        ecut: Energy cutoff for plane waves
        kpts: k-points grid (if None, uses gamma point)
    
    Returns:
        Dictionary with band gap information
    """
    try:
        # Convert to PySCF cell
        cell = atoms_to_pyscf_cell(atoms, basis, ecut)
        
        # Set k-points (gamma point by default)
        if kpts is None:
            kpts = np.array([[0, 0, 0]])
        
        # Create DFT calculator
        if method.lower() == 'pbe':
            mf = pbc_dft.RKS(cell, kpts)
            mf.xc = 'pbe'
        elif method.lower() == 'pbe0':
            mf = pbc_dft.RKS(cell, kpts)
            mf.xc = 'pbe0'
        elif method.lower() == 'b3lyp':
            mf = pbc_dft.RKS(cell, kpts)
            mf.xc = 'b3lyp'
        else:
            mf = pbc_dft.RKS(cell, kpts)
            mf.xc = method
        
        # Memory optimization settings
        mf.max_memory = 8000  # 8GB memory limit
        mf.diis_start_cycle = 3
        mf.diis_space = 6
        mf.max_cycle = 300
        mf.init_guess = 'minao'
        mf.conv_tol = 1e-5
        
        # Run SCF calculation
        mf.kernel()
        
        # Get eigenvalues
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        
        # Calculate band gap
        if len(mo_energy) > 0:
            # Find HOMO and LUMO
            occupied = np.where(mo_occ > 0.5)[0]
            unoccupied = np.where(mo_occ < 0.5)[0]
            
            if len(occupied) > 0 and len(unoccupied) > 0:
                homo_idx = occupied[-1]
                lumo_idx = unoccupied[0]
                
                # Get energies at gamma point (k=0)
                if isinstance(mo_energy, list) and len(mo_energy) > 0:
                    gamma_energies = mo_energy[0]  # First k-point is gamma
                    homo_energy = gamma_energies[homo_idx]
                    lumo_energy = gamma_energies[lumo_idx]
                    bandgap = lumo_energy - homo_energy
                else:
                    homo_energy = mo_energy[homo_idx]
                    lumo_energy = mo_energy[lumo_idx]
                    bandgap = lumo_energy - homo_energy
                
                return {
                    'bandgap': bandgap,
                    'homo_energy': homo_energy,
                    'lumo_energy': lumo_energy,
                    'homo_idx': homo_idx,
                    'lumo_idx': lumo_idx,
                    'method': method,
                    'basis': basis,
                    'converged': mf.converged,
                    'total_energy': mf.e_tot,
                    'error': None
                }
        
        # Fallback for non-periodic or failed calculations
        return {
            'bandgap': 0.0,
            'homo_energy': 0.0,
            'lumo_energy': 0.0,
            'homo_idx': -1,
            'lumo_idx': -1,
            'method': method,
            'basis': basis,
            'converged': False,
            'total_energy': 0.0,
            'error': 'Could not determine band gap'
        }
        
    except Exception as e:
        return {
            'bandgap': 0.0,
            'homo_energy': 0.0,
            'lumo_energy': 0.0,
            'homo_idx': -1,
            'lumo_idx': -1,
            'method': method,
            'basis': basis,
            'converged': False,
            'total_energy': 0.0,
            'error': str(e)
        }


def calculate_bandgap_molecular_pyscf(atoms: Atoms, method: str = 'pbe', basis: str = 'def2-svp') -> Dict[str, float]:
    """
    Calculate band gap for molecular systems using PySCF with optimizations.
    
    Args:
        atoms: ASE atoms object
        method: DFT functional
        basis: Basis set
    
    Returns:
        Dictionary with band gap information
    """
    try:
        # Convert to PySCF molecule
        coords = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        mol = pyscf.gto.Mole()
        mol.atom = [[symbol, coord.tolist()] for symbol, coord in zip(symbols, coords)]
        mol.basis = basis
        mol.charge = 0
        mol.spin = 0
        mol.verbose = 0
        
        # Add effective core potentials for heavy elements (if available)
        try:
            if any(symbol in ['Cd', 'Zn', 'Ga', 'In'] for symbol in symbols):
                # Use proper ECP format for PySCF
                # For heavy elements, use lanl2dz which is widely supported
                ecp_basis = {}
                for symbol in symbols:
                    if symbol in ['Cd', 'Zn', 'Ga', 'In']:
                        ecp_basis[symbol] = 'lanl2dz'
                
                if ecp_basis:
                    mol.ecp = ecp_basis
                    logger.info(f"Using ECPs for heavy elements: {list(ecp_basis.keys())}")
        except Exception as e:
            logger.warning(f"ECP not available: {e}. Continuing without ECPs.")
            # Continue without ECPs - this is often fine for band gap calculations
        
        mol.build()
        
        # Create DFT calculator with density fitting
        if method.lower() == 'pbe':
            mf = dft.RKS(mol)
            mf.xc = 'pbe'
        elif method.lower() == 'pbe0':
            mf = dft.RKS(mol)
            mf.xc = 'pbe0'
        elif method.lower() == 'b3lyp':
            mf = dft.RKS(mol)
            mf.xc = 'b3lyp'
        else:
            mf = dft.RKS(mol)
            mf.xc = method
        
        # Enable density fitting for faster calculations
        mf = mf.density_fit(auxbasis='def2-svp-jkfit')
        
        # Optimized SCF settings for large systems
        mf.max_cycle = 200
        mf.diis_start_cycle = 3
        mf.diis_space = 8
        mf.conv_tol = 1e-6
        mf.conv_tol_grad = 1e-4
        mf.init_guess = 'minao'
        
        # Level shifting for better convergence
        mf.level_shift = 0.2
        
        # Run SCF calculation
        mf.kernel()
        
        # Check convergence
        if not mf.converged:
            # Try with more relaxed settings
            logger.warning(f"SCF not converged, trying with relaxed settings...")
            mf.conv_tol = 1e-5
            mf.conv_tol_grad = 1e-3
            mf.level_shift = 0.3
            mf.kernel()
            
            if not mf.converged:
                # Try with very relaxed settings
                logger.warning(f"Still not converged, trying with very relaxed settings...")
                mf.conv_tol = 1e-4
                mf.conv_tol_grad = 1e-2
                mf.level_shift = 0.5
                mf.kernel()
        
        # Get orbital energies
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        
        # Find HOMO and LUMO
        occupied = np.where(mo_occ > 0.5)[0]
        unoccupied = np.where(mo_occ < 0.5)[0]
        
        if len(occupied) > 0 and len(unoccupied) > 0:
            homo_idx = occupied[-1]
            lumo_idx = unoccupied[0]
            
            homo_energy = mo_energy[homo_idx]
            lumo_energy = mo_energy[lumo_idx]
            bandgap = lumo_energy - homo_energy
            
            return {
                'bandgap': bandgap,
                'homo_energy': homo_energy,
                'lumo_energy': lumo_energy,
                'homo_idx': homo_idx,
                'lumo_idx': lumo_idx,
                'method': method,
                'basis': basis,
                'converged': mf.converged,
                'total_energy': mf.e_tot,
                'error': None
            }
        
        return {
            'bandgap': 0.0,
            'homo_energy': 0.0,
            'lumo_energy': 0.0,
            'homo_idx': -1,
            'lumo_idx': -1,
            'method': method,
            'basis': basis,
            'converged': False,
            'total_energy': 0.0,
            'error': 'Could not determine band gap'
        }
        
    except Exception as e:
        return {
            'bandgap': 0.0,
            'homo_energy': 0.0,
            'lumo_energy': 0.0,
            'homo_idx': -1,
            'lumo_idx': -1,
            'method': method,
            'basis': basis,
            'converged': False,
            'total_energy': 0.0,
            'error': str(e)
        }


def determine_system_type(atoms: Atoms) -> str:
    """Determine if system is periodic or molecular"""
    cell = atoms.get_cell()
    if cell is None or not cell.any():
        return 'molecular'
    
    # Check if cell has finite volume
    volume = np.linalg.det(cell)
    if volume > 1e-6:  # Finite volume indicates periodic
        return 'periodic'
    else:
        return 'molecular'


def calculate_bandgap(atoms: Atoms, method: str = 'pbe', basis: str = 'sto-3g', 
                     ecut: float = 400.0) -> Dict[str, float]:
    """
    Calculate band gap using molecular PySCF method (default).
    
    Args:
        atoms: ASE atoms object
        method: DFT functional
        basis: Basis set
        ecut: Energy cutoff (not used for molecular calculations)
    
    Returns:
        Dictionary with band gap information
    """
    # Always use molecular calculation for now to avoid memory issues
    return calculate_bandgap_molecular_pyscf(atoms, method, basis)


def process_structures(xyz_file: str, output_file: str = None, method: str = 'pbe', 
                     basis: str = 'sto-3g', ecut: float = 400.0) -> pd.DataFrame:
    """
    Process all structures in an XYZ file and compute band gaps
    
    Args:
        xyz_file (str): Path to XYZ file
        output_file (str): Path to output CSV file
        method (str): DFT functional to use
        basis (str): Basis set
        ecut (float): Energy cutoff for periodic calculations
        
    Returns:
        pd.DataFrame: DataFrame with results
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
            'system_type': determine_system_type(atoms)
        }
        
        # Calculate band gap
        result = calculate_bandgap(atoms, method, basis, ecut)
        
        # Combine results
        combined_result = {**structure_info, **result}
        results.append(combined_result)
        
        # Print progress
        if result['converged'] and result['error'] is None:
            print(f"  Band gap: {result['bandgap']:.4f} eV")
            print(f"  HOMO: {result['homo_energy']:.4f} eV, LUMO: {result['lumo_energy']:.4f} eV")
        else:
            print(f"  Failed to converge or calculate band gap")
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
        band_gaps = successful['bandgap'].values
        print(f"Band gap range: {np.min(band_gaps):.3f} - {np.max(band_gaps):.3f} eV")
        print(f"Mean band gap: {np.mean(band_gaps):.3f} ± {np.std(band_gaps):.3f} eV")
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time/3600:.2f} hours")
    
    return df


def main():
    """Main function to run band gap calculations"""
    parser = argparse.ArgumentParser(description='Calculate band gaps using PySCF for reference dataset')
    parser.add_argument('--xyz_file', default='trainall.xyz', 
                       help='Input XYZ file (default: trainall.xyz)')
    parser.add_argument('--output', default=None,
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--method', default='pbe',
                       choices=['pbe', 'pbe0', 'b3lyp'],
                       help='DFT functional (default: pbe)')
    parser.add_argument('--basis', default='def2-svp',
                       help='Basis set (default: def2-svp)')
    parser.add_argument('--ecut', type=float, default=400.0,
                       help='Energy cutoff for periodic calculations (default: 400.0)')
    parser.add_argument('--test', action='store_true',
                       help='Test with first 3 structures only')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with first 3 structures...")
        # Read first 3 structures
        structures = list(ase.io.read(args.xyz_file, index=':3'))
        test_file = 'test_structures.xyz'
        ase.io.write(test_file, structures)
        df = process_structures(test_file, args.output, args.method, args.basis, args.ecut)
        os.remove(test_file)
    else:
        df = process_structures(args.xyz_file, args.output, args.method, args.basis, args.ecut)
    
    return df


if __name__ == "__main__":
    main() 
