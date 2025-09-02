"""
ai generated
maybe try pbe0 soon

use pbe
plane wave basis set
8 k points in brillouin zone
Energy tolerance: 1×10⁻⁶ eV
Density tolerance: 1×10⁻⁶ electrons/Å³

FIXED: Stress calculation issue resolved by using correct GPAW configuration.
GPAW in plane-wave mode automatically computes forces and stress.
Two relaxation options available:
1. Atomic positions only (faster)
2. Full cell + atomic relaxation (more accurate)

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
    
    Note: GPAW in plane-wave mode automatically computes forces and stress.
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
        
        # Note: GPAW in plane-wave mode automatically computes forces and stress
        # No need to explicitly enable them
        
        return calculator
        
    except ImportError:
        print("GPAW not available. Install with: pip install gpaw")
        return None

def compute_gpaw_energy(atoms, calculator=None, relax_geometry=True, relax_cell=False):
    """
    Compute energy using GPAW calculator with optional geometry relaxation.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        Atomic structure
    calculator : gpaw.GPAW, optional
        GPAW calculator (will create one if not provided)
    relax_geometry : bool
        Whether to relax atomic positions
    relax_cell : bool
        Whether to relax cell parameters (requires stress calculation)
    
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
        
        if relax_geometry or relax_cell:
            print("Relaxing geometry...")
            
            if relax_cell:
                print("  - Relaxing both cell parameters and atomic positions")
                # Relax both cell parameters and atomic positions
                ucf = UnitCellFilter(atoms)
                opt = BFGS(ucf)
                opt.run(fmax=0.01, steps=100)  # Convergence criterion and max steps
                print(f"  - Cell + atomic relaxation converged after {opt.get_number_of_steps()} steps")
            else:
                print("  - Relaxing only atomic positions (keeping cell fixed)")
                # Relax only atomic positions, keeping cell parameters fixed
                opt = BFGS(atoms)
                opt.run(fmax=0.01, steps=100)  # Convergence criterion and max steps
                print(f"  - Atomic relaxation converged after {opt.get_number_of_steps()} steps")
        
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
    print("Choose relaxation method:")
    print("1. Atomic positions only (faster, no cell changes)")
    print("2. Full cell + atomic relaxation (slower, more accurate)")
    
    # For now, use the safer option (atomic positions only)
    # Change relax_cell=True if you want full cell relaxation
    relax_cell = False
    
    # Make a copy of atoms to preserve original structure
    atoms_relaxed = atoms.copy()
    
    energy = compute_gpaw_energy(atoms_relaxed, relax_geometry=True, relax_cell=relax_cell)
    if energy is not None:
        print(f"GPAW energy (relaxed): {energy:.6f} eV")
        
        # Show lattice parameter changes
        print("\nLattice parameter comparison:")
        print(f"Initial: a = {a:.3f} Å, c = {c:.3f} Å")
        relaxed_cell_params = atoms_relaxed.cell.cellpar()
        print(f"Relaxed: a = {relaxed_cell_params[0]:.3f} Å, c = {relaxed_cell_params[2]:.3f} Å")
        
        if relax_cell:
            print(f"Changes: Δa = {relaxed_cell_params[0] - a:+.3f} Å, Δc = {relaxed_cell_params[2] - c:+.3f} Å")
        else:
            print("Note: Cell parameters were kept fixed during relaxation")
        
        return energy, atoms_relaxed
    
    return None, None

def run_atomic_relaxation_only():
    """
    Run GPAW calculation with only atomic position relaxation (faster, safer).
    This avoids the stress calculation issue.
    """
    print("Running atomic position relaxation only...")
    atoms_relaxed = atoms.copy()
    
    energy = compute_gpaw_energy(atoms_relaxed, relax_geometry=True, relax_cell=False)
    if energy is not None:
        print(f"GPAW energy (atomic positions relaxed): {energy:.6f} eV")
        return energy, atoms_relaxed
    
    return None, None

def run_full_relaxation():
    """
    Run GPAW calculation with full cell + atomic relaxation (slower, more accurate).
    This requires stress calculation and may take longer.
    """
    print("Running full cell + atomic relaxation...")
    atoms_relaxed = atoms.copy()
    
    energy = compute_gpaw_energy(atoms_relaxed, relax_geometry=True, relax_cell=True)
    if energy is not None:
        print(f"GPAW energy (fully relaxed): {energy:.6f} eV")
        
        # Show lattice parameter changes
        relaxed_cell_params = atoms_relaxed.cell.cellpar()
        print(f"\nLattice parameter changes:")
        print(f"Δa = {relaxed_cell_params[0] - a:+.3f} Å, Δc = {relaxed_cell_params[2] - c:+.3f} Å")
        
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
    print("Choose calculation type:")
    print("  a) Atomic positions only (faster, safer)")
    print("  b) Full cell + atomic relaxation (slower, more accurate)")
    
    # Both options should work now with the fixed GPAW configuration
    choice = 'b'  # 'a' for atomic only, 'b' for full relaxation
    
    if choice == 'a':
        gpaw_energy, relaxed_atoms = run_atomic_relaxation_only()
    else:
        gpaw_energy, relaxed_atoms = run_full_relaxation()
    
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
