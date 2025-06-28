#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import time

from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
from config import get_system_config

def main():    
    # Equilibrium is around 2.25 Å, 
    distances = np.linspace(1.8, 3.5, 50)  

    system_config = get_system_config("Si2")

    ccsd_config = CalcConfig(
        method=CalcMethod.CCSD,
        basis="cc-pVTZ",  # Good quality basis for reference data
        spin=3  # Si2 is typically triplet in ground state
    )
        
    # Create calculator and generator
    calculator = GeneralCalculator(ccsd_config, system_config)
    generator = DissociationCurveGenerator(calculator)
    
    # Output file
    output_file = Path("results/curves/si2_ccsd_data.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output will be saved to: {output_file}")
    
    start_time = time.time()
    
    try:
        # Generate the dissociation curve
        curve_data = generator.generate_curve(
            distances=distances,
            save=True,
            filename=str(output_file)
        )
        
        elapsed = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"CCSD CALCULATION COMPLETED SUCCESSFULLY!")
        print(f"Time elapsed: {elapsed/3600:.2f} hours")
        print(f"Data saved to: {output_file}")
        
        # Print some statistics
        energies = curve_data['Energy'].values
        min_idx = np.argmin(energies)
        min_energy = energies[min_idx]
        min_distance = distances[min_idx]
        
        print(f"  Minimum energy: {min_energy:.8f} Hartree")
        print(f"  Equilibrium distance: {min_distance:.3f} Å")
                
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"ERROR: CCSD calculation failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    main() 
