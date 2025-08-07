"""
ai generated
maybe try pbe0 soon

use pbe
plane wave basis set
8 k points in brillouin zone
Energy tolerance: 1×10⁻⁶ eV
Density tolerance: 1×10⁻⁶ electrons/Å³

"""


import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from tblite.ase import TBLite

# ===================================================================
# 1. Define the Crystal Structure with ASE
# ===================================================================
# CdS wurtzite structure parameters (experimental values)
a = 4.139
c = 6.720
atoms = bulk('CdS', 'wurtzite', a=a, c=c)

# ===================================================================
# 2. GPAW Implementation (Recommended for Solids)
# ===================================================================
def create_gpaw_calculator():
    """
    Create GPAW calculator for high-accuracy solid-state calculations.
    GPAW is excellent for periodic systems like CdS wurtzite.
    """
    try:
        import gpaw
        
        # GPAW calculator for periodic systems
        calculator = gpaw.GPAW(
            mode='pw',              # Plane-wave mode (best for solids)
            xc='PBE',              # Exchange-correlation functional
            kpts=(2, 2, 2),        # K-point mesh
            charge=0,              # Total charge
            spinpol=False,         # No spin polarization
            txt='gpaw_output.txt', # Output file
            maxiter=100,           # SCF iterations
            convergence={'energy': 1e-6, 'density': 1e-6},  # Convergence criteria
        )
        
        return calculator
        
    except ImportError:
        print("GPAW not available. Install with: pip install gpaw")
        return None

def compute_gpaw_energy(atoms, calculator=None):
    """
    Compute energy using GPAW calculator.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        Atomic structure
    calculator : gpaw.GPAW, optional
        GPAW calculator (will create one if not provided)
    
    Returns:
    --------
    float : Energy in eV
    """
    if calculator is None:
        calculator = create_gpaw_calculator()
        if calculator is None:
            return None
    
    try:
        # Set calculator
        atoms.calc = calculator
        
        # Compute energy
        energy = atoms.get_potential_energy()
        
        return energy
        
    except Exception as e:
        print(f"GPAW calculation failed: {e}")
        return None

def example_gpaw_calculation():
    """
    Example GPAW calculation for CdS wurtzite.
    """
    energy = compute_gpaw_energy(atoms)
    if energy is not None:
        print(f"GPAW energy: {energy:.6f} eV")
    return energy

# ===================================================================
# 4. Main Function
# ===================================================================
def main():
    """
    Main function to compute reference energy for CdS wurtzite.
    """
    print("=" * 50)
    print("Reference Energy Calculation for CdS Wurtzite")
    print("=" * 50)
    
    print(f"Structure: CdS wurtzite")
    print(f"Lattice parameters: a = {a:.3f} Å, c = {c:.3f} Å")
    print(f"Unit cell: {len(atoms)} atoms")
    
    # GPAW calculation (recommended)
    print("\n1. GPAW Calculation (recommended):")
    gpaw_energy = example_gpaw_calculation()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if gpaw_energy is not None:
        print(f"GPAW (recommended): {gpaw_energy:.6f} eV")
    else:
        print("GPAW: Failed")
        
    # Recommendation
    if gpaw_energy is not None:
        print(f"\nRecommended reference energy: {gpaw_energy:.6f} eV")
        print("Use this value for TBLite parameter optimization.")
    
    return gpaw_energy

if __name__ == "__main__":
    main()
