#!/usr/bin/env python3
"""
Test script for band gap calculator
Tests with a single CdS structure first
"""

import sys
import os
sys.path.append('src')

from band_gap_calculator import BandGapCalculator
import ase.io
import numpy as np

def test_single_structure():
    """Test band gap calculation with a single structure"""
    print("Testing band gap calculator with single CdS structure...")
    
    # Read first structure from training data
    structures = list(ase.io.read('trainall.xyz', index=':'))
    if len(structures) == 0:
        print("No structures found in trainall.xyz")
        return
    
    # Take first structure
    atoms = structures[0]
    print(f"Testing with structure: {atoms.get_chemical_formula()}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell volume: {atoms.get_volume():.2f} Å³")
    print(f"Cell parameters:")
    print(atoms.cell)
    
    # Initialize calculator with minimal settings for testing
    calculator = BandGapCalculator(
        functional='PBE',
        basis='gth-dzvp',
        verbose=1  # Show some output for debugging
    )
    
    # Calculate band gap
    print("\nCalculating band gap...")
    result = calculator.calculate_band_gap(atoms)
    
    # Print results
    print("\n=== BAND GAP RESULTS ===")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return result

def test_multiple_structures(n_structures=5):
    """Test with multiple structures"""
    print(f"\nTesting with {n_structures} structures...")
    
    # Read structures
    structures = list(ase.io.read('trainall.xyz', index=':'))
    structures = structures[:n_structures]
    
    # Initialize calculator
    calculator = BandGapCalculator(
        functional='PBE',
        basis='gth-dzvp',
        verbose=0
    )
    
    results = []
    for i, atoms in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}")
        result = calculator.calculate_band_gap(atoms)
        result['structure_id'] = i
        results.append(result)
    
    # Print summary
    print("\n=== SUMMARY ===")
    successful = [r for r in results if r['error'] is None]
    failed = [r for r in results if r['error'] is not None]
    
    print(f"Successful calculations: {len(successful)}")
    print(f"Failed calculations: {len(failed)}")
    
    if successful:
        band_gaps = [r['band_gap'] for r in successful]
        print(f"Band gap range: {min(band_gaps):.3f} - {max(band_gaps):.3f} eV")
        print(f"Mean band gap: {np.mean(band_gaps):.3f} ± {np.std(band_gaps):.3f} eV")
    
    if failed:
        print("\nFailed calculations:")
        for r in failed:
            print(f"  Structure {r['structure_id']}: {r['error']}")

if __name__ == "__main__":
    # Test single structure first
    result = test_single_structure()
    
    # If successful, test multiple structures
    if result and result['error'] is None:
        test_multiple_structures(5)
    else:
        print("Single structure test failed, not proceeding to multiple structures") 
