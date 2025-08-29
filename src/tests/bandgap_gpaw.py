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
import os
import time
import tempfile
import shutil
from pathlib import Path
from gpaw import GPAW, PW, FermiDirac

# Configuration - edit these as needed
XYZ_FILE = 'trainall.xyz'
OUTPUT_FILE = 'bands_pbe_gpaw.csv'
METHOD = 'PBE'  # Options: 'PBE', 'PBE0', 'HSE06'
TEST_MODE = False  # Set to False for full dataset
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
    
    try:
        # Perform SCF calculation to get converged wavefunctions
        energy = atoms.get_potential_energy()
        logger.info(f"  SCF converged with energy: {energy:.6f} eV")
        
        # Get eigenvalues at gamma point
        eigenvalues = calc.get_eigenvalues(kpt=0)
        nelec = calc.get_number_of_electrons()
        
        # Calculate band structure (optional, can be commented out for speed)
        # bs = calc.band_structure()
        # bs.plot(filename='bandstructure.png', show=False, emax=10.0)
        
        # Find HOMO and LUMO
        homo_idx = int(nelec // 2) - 1
        lumo_idx = int(nelec // 2)
        
        if homo_idx >= 0 and lumo_idx < len(eigenvalues):
            homo_energy = eigenvalues[homo_idx]
            lumo_energy = eigenvalues[lumo_idx]
            band_gap = lumo_energy - homo_energy
            return band_gap, homo_energy, lumo_energy, None  # Return None for bs to avoid issues
        else:
            logger.warning(f"  Invalid HOMO/LUMO indices: {homo_idx}, {lumo_idx}, nelec={nelec}, n_eigenvalues={len(eigenvalues)}")
            return float('nan'), float('nan'), float('nan'), None
            
    except Exception as e:
        logger.error(f"  SCF calculation failed: {e}")
        return float('nan'), float('nan'), float('nan'), None


def save_checkpoint(results, output_file, checkpoint_file):
    """Save results to checkpoint file atomically"""
    try:
        # Create temporary file
        temp_file = checkpoint_file + '.tmp'
        
        # Save to temporary file
        df = pd.DataFrame(results)
        df.to_csv(temp_file, index=False)
        
        # Atomic move
        shutil.move(temp_file, checkpoint_file)
        
        # Also save to final output file
        df.to_csv(output_file, index=False)
        
        logger.info(f"  Checkpoint saved: {len(results)} structures completed")
        return True
    except Exception as e:
        logger.error(f"  Failed to save checkpoint: {e}")
        return False


def load_checkpoint(checkpoint_file):
    """Load existing checkpoint if available"""
    if os.path.exists(checkpoint_file):
        try:
            df = pd.read_csv(checkpoint_file)
            logger.info(f"Loaded checkpoint with {len(df)} completed structures")
            return df.to_dict('records')
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return []


def get_completed_structure_ids(results):
    """Get set of completed structure IDs"""
    return {r['structure_id'] for r in results if pd.notna(r['band_gap'])}

def main():
    # Use global configuration variables
    xyz_file = XYZ_FILE
    output_file = OUTPUT_FILE
    method = METHOD
    test_mode = TEST_MODE
    checkpoint_file = output_file + '.checkpoint'
    
    # Read structures
    if test_mode:
        structures = list(ase.io.read(xyz_file, index=':3'))
        logger.info(f"Testing with {len(structures)} structures")
    else:
        structures = list(ase.io.read(xyz_file, index=':'))
        logger.info(f"Processing {len(structures)} structures")
    
    # Load existing results if checkpointing is enabled
    results = load_checkpoint(checkpoint_file)
    completed_ids = get_completed_structure_ids(results)
    logger.info(f"Resuming from checkpoint: {len(completed_ids)} structures already completed")
    
    # Calculate remaining structures
    start_time = time.time()
    total_structures = len(structures)
    
    for i, atoms in enumerate(structures):
        # Skip if already completed
        if i in completed_ids:
            logger.info(f"Structure {i+1}: {atoms.get_chemical_formula()} (already completed)")
            continue
            
        logger.info(f"Structure {i+1}/{total_structures}: {atoms.get_chemical_formula()}")
        structure_start_time = time.time()
        
        try:
            band_gap, homo, lumo, bs = calculate_bandgap(atoms, args.method)
            
            if pd.notna(band_gap):
                logger.info(f"  Band gap: {band_gap:.3f} eV")
            else:
                logger.warning(f"  Failed to calculate band gap")
            
            result = {
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'band_gap': band_gap,
                'homo_energy': homo,
                'lumo_energy': lumo,
                'bs': 'calculated' if bs is not None else 'None',
                'calculation_time': time.time() - structure_start_time
            }
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            result = {
                'structure_id': i,
                'formula': atoms.get_chemical_formula(),
                'band_gap': float('nan'),
                'homo_energy': float('nan'),
                'lumo_energy': float('nan'),
                'bs': 'failed',
                'calculation_time': time.time() - structure_start_time,
                'error': str(e)
            }
        
        # Add to results
        results.append(result)
        
        # Save checkpoint after each calculation
        if checkpoint_file and not args.no_checkpoint:
            save_checkpoint(results, args.output, checkpoint_file)
        
        # Progress update
        elapsed = time.time() - start_time
        completed = len([r for r in results if pd.notna(r['band_gap'])])
        remaining = total_structures - completed
        if remaining > 0:
            avg_time_per_structure = elapsed / completed if completed > 0 else 0
            estimated_remaining_time = avg_time_per_structure * remaining
            logger.info(f"  Progress: {completed}/{total_structures} completed ({completed/total_structures*100:.1f}%)")
            logger.info(f"  Estimated time remaining: {estimated_remaining_time/3600:.1f} hours")
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    valid_gaps = df['band_gap'].dropna()
    if len(valid_gaps) > 0:
        logger.info(f"Mean band gap: {valid_gaps.mean():.3f} ± {valid_gaps.std():.3f} eV")
        logger.info(f"Total calculation time: {(time.time() - start_time)/3600:.1f} hours")
    else:
        logger.info("No valid band gaps calculated")

if __name__ == "__main__":
    main()
