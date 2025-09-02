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
# GaN wurtzite structure parameters (experimental values)
a = 3.19
c = 5.19
atoms = bulk('GaN', 'wurtzite', a=a, c=c)

# ===================================================================
# 2. GPAW Implementation (Recommended for Solids)
# ===================================================================
def create_gpaw_calculator():
    """
    Create GPAW calculator for high-accuracy solid-state calculations.
    GPAW is excellent for periodic systems like GaN wurtzite.
    """
    try:
        import gpaw
        
        # GPAW calculator for periodic systems
        calculator = gpaw.GPAW(
            mode='pw',              # Plane-wave mode (best for solids)
            xc='HSE06',              # Exchange-correlation functional
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

def compute_gpaw_energy(atoms, calculator=None, relax_geometry=True):
    """
    Compute energy using GPAW calculator with optional geometry relaxation.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        Atomic structure
    calculator : gpaw.GPAW, optional
        GPAW calculator (will create one if not provided)
    relax_geometry : bool
        Whether to relax the geometry before computing energy
    
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
        
        if relax_geometry:
            print("Relaxing geometry...")
            # Relax both cell parameters and atomic positions
            ucf = UnitCellFilter(atoms)
            opt = BFGS(ucf)
            opt.run(fmax=0.01, steps=100)  # Convergence criterion and max steps
            print(f"Geometry relaxation converged after {opt.get_number_of_steps()} steps")
        
        # Compute energy
        energy = atoms.get_potential_energy()
        
        return energy
        
    except Exception as e:
        print(f"GPAW calculation failed: {e}")
        return None

def example_gpaw_calculation():
    """
    Example GPAW calculation for GaN wurtzite with geometry relaxation.
    """
    # Make a copy of atoms to preserve original structure
    atoms_relaxed = atoms.copy()
    
    energy = compute_gpaw_energy(atoms_relaxed, relax_geometry=True)
    if energy is not None:
        print(f"GPAW energy (relaxed): {energy:.6f} eV")
        
        # Show lattice parameter changes
        print("\nLattice parameter comparison:")
        print(f"Initial: a = {a:.3f} Å, c = {c:.3f} Å")
        relaxed_cell = atoms_relaxed.cell.cellpar()
        print(f"Relaxed: a = {relaxed_cell[0]:.3f} Å, c = {relaxed_cell[2]:.3f} Å")
        print(f"Changes: Δa = {relaxed_cell[0] - a:+.3f} Å, Δc = {relaxed_cell[2] - c:+.3f} Å")
        
        return energy, atoms_relaxed
    
    return None, None




# ===================================================================
# 4. Main Function
# ===================================================================
def main():
    """
    Main function to compute reference energy for GaN wurtzite.
    """
    print("=" * 50)
    print("Reference Energy Calculation for GaN Wurtzite")
    print("=" * 50)
    
    print(f"Structure: GaN wurtzite")
    print(f"Lattice parameters: a = {a:.3f} Å, c = {c:.3f} Å")
    print(f"Unit cell: {len(atoms)} atoms")
    
    # GPAW calculation (recommended)
    print("\n1. GPAW Calculation (recommended):")
    gpaw_energy, relaxed_atoms = example_gpaw_calculation()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if gpaw_energy is not None:
        print(f"GPAW (recommended): {gpaw_energy:.6f} eV")
        
        if relaxed_atoms is not None:
            relaxed_cell = relaxed_atoms.cell.cellpar()
            print(f"Relaxed lattice parameters:")
            print(f"  a = {relaxed_cell[0]:.3f} Å, b = {relaxed_cell[1]:.3f} Å, c = {relaxed_cell[2]:.3f} Å")
            print(f"  α = {relaxed_cell[3]:.1f}°, β = {relaxed_cell[4]:.1f}°, γ = {relaxed_cell[5]:.1f}°")
            
            # Save relaxed structure
            relaxed_filename = "GaN_relaxed.xyz"
            relaxed_atoms.write(relaxed_filename)
            print(f"Relaxed structure saved to: {relaxed_filename}")
    else:
        print("GPAW: Failed")
        
    # Recommendation
    if gpaw_energy is not None:
        print(f"\nRecommended reference energy: {gpaw_energy:.6f} eV")
        print("Use this value for TBLite parameter optimization.")
        print("Note: This energy is for the relaxed geometry, not the initial structure.")
    
    return gpaw_energy

if __name__ == "__main__":
    main()
