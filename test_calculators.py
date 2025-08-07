#!/usr/bin/env python3

print("Testing ASE calculator availability...")

try:
    import ase
    print(f"ASE version: {ase.__version__}")
except ImportError:
    print("ASE not available")
    exit(1)

# Test GPAW
try:
    import ase.calculators.gpaw
    print("✓ GPAW calculator available")
    gpaw_available = True
except ImportError:
    print("✗ GPAW calculator not available")
    gpaw_available = False

# Test PySCF
try:
    import ase.calculators.pyscf
    print("✓ PySCF calculator available")
    pyscf_available = True
except ImportError:
    print("✗ PySCF calculator not available")
    pyscf_available = False

# Test VASP
try:
    import ase.calculators.vasp
    print("✓ VASP calculator available")
    vasp_available = True
except ImportError:
    print("✗ VASP calculator not available")
    vasp_available = False

# Test Quantum ESPRESSO
try:
    import ase.calculators.espresso
    print("✓ Quantum ESPRESSO calculator available")
    espresso_available = True
except ImportError:
    print("✗ Quantum ESPRESSO calculator not available")
    espresso_available = False

# Test TBLite
try:
    import tblite.ase
    print("✓ TBLite calculator available")
    tblite_available = True
except ImportError:
    print("✗ TBLite calculator not available")
    tblite_available = False

print("\nSummary:")
print(f"GPAW: {gpaw_available}")
print(f"PySCF: {pyscf_available}")
print(f"VASP: {vasp_available}")
print(f"Quantum ESPRESSO: {espresso_available}")
print(f"TBLite: {tblite_available}")
