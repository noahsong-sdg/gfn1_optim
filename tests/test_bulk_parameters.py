#!/usr/bin/env python3
"""
Test optimized TBLite parameters on bulk materials datasets.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from calculators.bulk_calculator import BulkCalculator
from config import get_system_config
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Test optimized parameters on bulk datasets')
    parser.add_argument('param_file', help='Path to optimized parameter file')
    parser.add_argument('xyz_file', help='Path to test XYZ file')
    parser.add_argument('--max-structures', type=int, default=50,
                       help='Maximum number of structures to test (default: 50)')
    parser.add_argument('--output', help='Output CSV file for detailed results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.param_file).exists():
        print(f"Error: Parameter file not found: {args.param_file}")
        return
    
    if not Path(args.xyz_file).exists():
        print(f"Error: XYZ file not found: {args.xyz_file}")
        return
    
    print(f"Testing parameters: {args.param_file}")
    print(f"Test dataset: {args.xyz_file}")
    print(f"Max structures: {args.max_structures}")
    
    # Create bulk calculator
    system_config = get_system_config("BulkMaterials")  # Use any bulk config
    bulk_calc = BulkCalculator(
        param_file=args.param_file,
        method="gfn1",
        electronic_temperature=400.0,
        charge=0.0,
        spin=0
    )
    
    # Test parameters
    metrics = bulk_calc.test_parameters_on_dataset(
        param_file=args.param_file,
        xyz_file=args.xyz_file,
        max_structures=args.max_structures
    )
    
    # Print results
    print("\n=== TEST RESULTS ===")
    print(f"RMSE: {metrics['rmse']:.6f} eV")
    print(f"MAE: {metrics['mae']:.6f} eV")
    print(f"Max Error: {metrics['max_error']:.6f} eV")
    print(f"Bias: {metrics['bias']:.6f} eV")
    print(f"Structures tested: {metrics['n_structures']}")
    
    # Save detailed results if requested
    if args.output:
        # Load structures and get detailed comparison
        structures = bulk_calc.load_structures_from_xyz(args.xyz_file, args.max_structures)
        reference_energies = bulk_calc.extract_reference_energies(structures)
        calculated_energies = bulk_calc.calculate_energies(structures)
        
        # Create detailed results DataFrame
        results = []
        for i, (ref, calc) in enumerate(zip(reference_energies, calculated_energies)):
            if not (np.isnan(ref) or np.isnan(calc)):
                results.append({
                    'structure_id': i,
                    'reference_energy': ref,
                    'calculated_energy': calc,
                    'error': calc - ref,
                    'abs_error': abs(calc - ref)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
