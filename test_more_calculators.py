#!/usr/bin/env python3

print("Testing additional DFT calculator availability...")

# Test PySCF directly (not ASE interface)
try:
    import pyscf
    print(f"✓ PySCF available (version: {pyscf.__version__})")
    pyscf_available = True
except ImportError:
    print("✗ PySCF not available")
    pyscf_available = False

# Test GPAW directly
try:
    import gpaw
    print("✓ GPAW available")
    gpaw_available = True
except ImportError:
    print("✗ GPAW not available")
    gpaw_available = False

# Test ORCA (if available)
try:
    import ase.calculators.orca
    print("✓ ORCA calculator available")
    orca_available = True
except ImportError:
    print("✗ ORCA calculator not available")
    orca_available = False

# Test NWChem
try:
    import ase.calculators.nwchem
    print("✓ NWChem calculator available")
    nwchem_available = True
except ImportError:
    print("✗ NWChem calculator not available")
    nwchem_available = False

# Test CP2K
try:
    import ase.calculators.cp2k
    print("✓ CP2K calculator available")
    cp2k_available = True
except ImportError:
    print("✗ CP2K calculator not available")
    cp2k_available = False

# Test FHI-aims
try:
    import ase.calculators.aims
    print("✓ FHI-aims calculator available")
    aims_available = True
except ImportError:
    print("✗ FHI-aims calculator not available")
    aims_available = False

print("\nSummary:")
print(f"PySCF (direct): {pyscf_available}")
print(f"GPAW (direct): {gpaw_available}")
print(f"ORCA: {orca_available}")
print(f"NWChem: {nwchem_available}")
print(f"CP2K: {cp2k_available}")
print(f"FHI-aims: {aims_available}")
