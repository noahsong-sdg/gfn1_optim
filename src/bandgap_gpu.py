#!/usr/bin/env python3
"""
GPU-accelerated band gap calculation script for reference dataset generation.
Uses PySCF with GPU acceleration for high-quality DFT calculations with proper band gap extraction.
Optimized for AMD GPUs and large-scale processing of hundreds of structures.
"""

import sys
import os
import json
import argparse
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator

# PySCF imports with GPU support
import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.dft
import pyscf.pbc.scf
import pyscf.pbc.tools
from pyscf import dft, scf
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import scf as pbc_scf

# GPU-specific imports
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Warning: CuPy not available. Using CPU-only mode.")

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_gpu_environment():
    """Setup GPU environment for AMD GPUs"""
    if HAS_CUPY:
        # Set environment variables for AMD GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        os.environ['HIP_VISIBLE_DEVICES'] = '0'   # AMD ROCm
        
        # Configure PySCF for GPU
        pyscf.lib.num_threads(1)  # Single thread for GPU calculations
        
        logger.info("GPU environment configured for AMD GPU")
        return True
    else:
        logger.warning("CuPy not available, using CPU-only mode")
        return False


def calculate_bandgap_gpu_optimized(atoms: Atoms, method: str = 'pbe', basis: str = 'def2-svp') -> Dict[str, float]:
    """
    Calculate band gap using GPU-accelerated PySCF with optimizations.
    
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
            if any(symbol in ['Cd', 'Zn', 'Ga', 'In', 'S', 'Se', 'Te'] for symbol in symbols):
                # Use proper ECP format for PySCF
                # For heavy elements, use lanl2dz which is widely supported
                ecp_basis = {}
                for symbol in symbols:
                    if symbol in ['Cd', 'Zn', 'Ga', 'In', 'S', 'Se', 'Te']:
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
        
        # GPU acceleration if available
        if HAS_CUPY:
            # Use GPU-accelerated version
            mf = mf.to_gpu()
        
        # Optimized SCF settings for GPU calculations
        mf.max_cycle = 150  # Reduced for GPU
        mf.diis_start_cycle = 3
        mf.diis_space = 6
        mf.conv_tol = 1e-6
        mf.conv_tol_grad = 1e-4
        mf.init_guess = 'minao'
        
        # Level shifting for better convergence
        mf.level_shift = 0.2
        
        # Run SCF calculation
        mf.kernel()
        
        # Check convergence with progressive relaxation
        if not mf.converged:
            logger.warning(f"SCF not converged, trying with relaxed settings...")
            mf.conv_tol = 1e-5
            mf.conv_tol_grad = 1e-3
            mf.level_shift = 0.3
            mf.kernel()
            
            if not mf.converged:
                logger.warning(f"Still not converged, trying with very relaxed settings...")
                mf.conv_tol = 1e-4
                mf.conv_tol_grad = 1e-2
                mf.level_shift = 0.5
                mf.kernel()
        
        # Get orbital energies
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        
        # Convert to CPU if on GPU
        if HAS_CUPY and hasattr(mo_energy, 'get'):
            mo_energy = mo_energy.get()
            mo_occ = mo_occ.get()
        
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


def process_single_structure(args):
    """Process a single structure (for parallel processing)"""
    i, atoms, method, basis = args
    
    try:
        logger.info(f"Processing structure {i+1}: {atoms.get_chemical_formula()}")
        
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
        start_time = time.time()
        result = calculate_bandgap_gpu_optimized(atoms, method, basis)
        calc_time = time.time() - start_time
        
        # Combine results
        combined_result = {**structure_info, **result, 'calculation_time': calc_time}
        
        # Print progress
        if result['converged'] and result['error'] is None:
            logger.info(f"  Structure {i+1}: Band gap = {result['bandgap']:.4f} eV (time: {calc_time:.1f}s)")
        else:
            logger.warning(f"  Structure {i+1}: Failed to converge (time: {calc_time:.1f}s)")
            if result['error']:
                logger.warning(f"    Error: {result['error']}")
        
        return combined_result
        
    except Exception as e:
        logger.error(f"Error processing structure {i+1}: {e}")
        return {
            'structure_id': i,
            'formula': atoms.get_chemical_formula() if hasattr(atoms, 'get_chemical_formula') else 'Unknown',
            'n_atoms': len(atoms),
            'volume': 0.0,
            'cell_a': 0.0,
            'cell_b': 0.0,
            'cell_c': 0.0,
            'bandgap': 0.0,
            'homo_energy': 0.0,
            'lumo_energy': 0.0,
            'homo_idx': -1,
            'lumo_idx': -1,
            'method': method,
            'basis': basis,
            'converged': False,
            'total_energy': 0.0,
            'error': str(e),
            'calculation_time': 0.0
        }


def process_structures_gpu(xyz_file: str, output_file: str = None, method: str = 'pbe', 
                         basis: str = 'def2-svp', n_workers: int = None) -> pd.DataFrame:
    """
    Process all structures in an XYZ file using GPU-accelerated calculations
    
    Args:
        xyz_file (str): Path to XYZ file
        output_file (str): Path to output CSV file
        method (str): DFT functional to use
        basis (str): Basis set
        n_workers (int): Number of parallel workers (default: number of CPU cores)
        
    Returns:
        pd.DataFrame: DataFrame with results
    """
    logger.info(f"Reading structures from {xyz_file}...")
    structures = list(ase.io.read(xyz_file, index=':'))
    logger.info(f"Found {len(structures)} structures")
    
    # Setup GPU environment
    gpu_available = setup_gpu_environment()
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers for GPU memory
    
    logger.info(f"Using {n_workers} parallel workers")
    if gpu_available:
        logger.info("GPU acceleration enabled")
    else:
        logger.info("Using CPU-only mode")
    
    results = []
    start_time = time.time()
    
    # Prepare arguments for parallel processing
    args_list = [(i, atoms, method, basis) for i, atoms in enumerate(structures)]
    
    # Process structures in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_idx = {executor.submit(process_single_structure, args): args[0] 
                        for args in args_list}
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                
                # Save intermediate results every 20 structures
                if len(results) % 20 == 0:
                    df_temp = pd.DataFrame(results)
                    temp_file = f"temp_gpu_results_{len(results)}.csv"
                    df_temp.to_csv(temp_file, index=False)
                    logger.info(f"Saved intermediate results to {temp_file}")
                    
            except Exception as e:
                logger.error(f"Error processing structure {idx+1}: {e}")
    
    # Sort results by structure_id to maintain order
    results.sort(key=lambda x: x['structure_id'])
    
    # Create final DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if output_file is None:
        output_file = f"band_gap_gpu_results_{Path(xyz_file).stem}_{method}_{basis}.csv"
    
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    successful = df[df['converged'] == True]
    failed = df[df['converged'] == False]
    
    total_time = time.time() - start_time
    avg_time = total_time / len(structures) if structures else 0
    
    logger.info(f"\n=== GPU ACCELERATED SUMMARY ===")
    logger.info(f"Total structures: {len(structures)}")
    logger.info(f"Successful calculations: {len(successful)}")
    logger.info(f"Failed calculations: {len(failed)}")
    logger.info(f"Success rate: {len(successful)/len(structures)*100:.1f}%")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Average time per structure: {avg_time:.1f} seconds")
    
    if len(successful) > 0:
        band_gaps = successful['bandgap'].values
        logger.info(f"Band gap range: {np.min(band_gaps):.3f} - {np.max(band_gaps):.3f} eV")
        logger.info(f"Mean band gap: {np.mean(band_gaps):.3f} ± {np.std(band_gaps):.3f} eV")
    
    return df


def main():
    """Main function to run GPU-accelerated band gap calculations"""
    parser = argparse.ArgumentParser(description='Calculate band gaps using GPU-accelerated PySCF')
    parser.add_argument('--xyz_file', default='trainall.xyz', 
                       help='Input XYZ file (default: trainall.xyz)')
    parser.add_argument('--output', default=None,
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--method', default='pbe',
                       choices=['pbe', 'pbe0', 'b3lyp'],
                       help='DFT functional (default: pbe)')
    parser.add_argument('--basis', default='def2-svp',
                       help='Basis set (default: def2-svp)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--test', action='store_true',
                       help='Test with first 5 structures only')
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in test mode with first 5 structures...")
        # Read first 5 structures
        structures = list(ase.io.read(args.xyz_file, index=':5'))
        test_file = 'test_structures_gpu.xyz'
        ase.io.write(test_file, structures)
        df = process_structures_gpu(test_file, args.output, args.method, args.basis, args.workers)
        os.remove(test_file)
    else:
        df = process_structures_gpu(args.xyz_file, args.output, args.method, args.basis, args.workers)
    
    return df


if __name__ == "__main__":
    main() 
