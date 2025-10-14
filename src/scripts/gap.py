import qcelemental as qcel
import tblite.interface as tb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
# Enable TBLite stdout for debugging
os.environ['TBLITE_PRINT_STDOUT'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ase import Atoms
from ase.build import bulk
from tblite.ase import TBLite
from ase.optimize import BFGS, FIRE
from ase.filters import UnitCellFilter

from calculators.tblite_ase_calculator import TBLiteASECalculator
from ase.io import read, write

atoms = read('cspbi3.cif')
atoms.calc = TBLite(method='GFN1-xTB', kpts={'size': (2, 2, 2)})
ucf = UnitCellFilter(atoms)
opt = FIRE(ucf) 
opt.run(fmax=0.01) 

# ------------------------- P2 ------------------------------- #
from ase.dft.kpoints import monkhorst_pack
relaxed_cell = atoms.get_cell()
relaxed_positions = atoms.get_positions()
atomic_numbers = atoms.get_atomic_numbers()

direct_calc = Calculator(
    method="GFN1-xTB",
    numbers=atomic_numbers,
    positions=relaxed_positions,
    lattice=relaxed_cell
)

k_grid = monkhorst_pack((4, 4, 4))
results = direct_calc.singlepoint(kpts=k_grid)

# Extract energies and occupations
energies = results.get("orbital-energies")[0]      # Shape: (nkpts, nbands)
occupations = results.get("orbital-occupations")[0] # Shape: (nkpts, nbands)
print("Calculation finished.")

# Use occupations to find the true VBM and CBM across all k-points
vbm = np.max(energies[occupations > 0.1])  # Use a threshold for occupation
cbm = np.min(energies[occupations < 0.1])  # Use a threshold for unoccupation
band_gap = cbm - vbm

print("\n--- Final Results ---")
print(f"Valence Band Maximum (VBM): {vbm:.4f} eV")
print(f"Conduction Band Minimum (CBM): {cbm:.4f} eV")
print(f"Band Gap: {band_gap:.4f} eV")
