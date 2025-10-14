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
import torch

from calculators.tblite_ase_calculator import TBLiteASECalculator
from ase.io import read, write
from ase.optimize import LBFGS

# ------------------------------------------------------------------------------------------------------ #
# unit cell
# atoms = bulk("CdS", "wurtzite", a = 4.17, c = 6.78)
# syms = atoms.get_chemical_symbols()
# positions = atoms.get_positions()
# molecule = qcel.models.Molecule(symbols=syms, geometry=positions) 

# cspbi3
# molecule = qcel.models.Molecule.from_file("cspbi3.xyz")
# syms = molecule.symbols # needed to convett to atoms
# geometry_angstroms = molecule.geometry * 0.529177210903 # needed to convert to atoms

# supercell 
# molecule = qcel.models.Molecule.from_file("pure_72_test.xyz")
# xtb = tb.Calculator("GFN1-xTB", molecule.atomic_numbers, molecule.geometry)
# xtb.set("temperature", 0.0002) # in hartrees. going to kelvin is by dividing by 3.167e-6
# results = xtb.singlepoint()
# orbital_energies = results["orbital-energies"]
# orbital_occupations = results["orbital-occupations"]

# occupied_indices = np.where(orbital_occupations > 1.8)[0]
# homo_index = occupied_indices[-1]  
# print("orbital occupancies: ", orbital_occupations[np.where(orbital_occupations > 0.5)])
# print("homo occupation: ", orbital_occupations[homo_index])
# print("homo energy: ", orbital_energies[homo_index])
# unoccupied_indices = np.where(orbital_occupations < 0.1)[0]
# lumo_index = unoccupied_indices[0]  
# print("orbital unoccupancies: ", orbital_occupations[np.where(orbital_occupations <= 0.5)])
# print("lumo occupation: ", orbital_occupations[lumo_index])
# print("lumo energy: ", orbital_energies[lumo_index])
# gap = (orbital_energies[lumo_index] - orbital_energies[homo_index])

# print(results["energy"] * 27.2114)
# print(gap)
# ------------------------------------------------------------------------------------------------------ #
# CsPbI3

from ase.io import read
atoms = read('cspbi3.cif') 
# atoms.calc = TBLiteASECalculator(
#     param_file="config/gfn1-base.toml",  
#     method="gfn1",
#     electronic_temperature=150.0,
#     charge=0.0,
#     spin=0
# )
atoms.calc = TBLite(method="GFN1-xTB", 
                    electronic_temperature=100.0, 
                    max_iterations=2000,
                    initial_guess="sad",
                    accuracy=1.0,
                    kpts={'size': (4, 4, 4), 'gamma': True})
# has no attribute get_ibz_k_points
ucf = UnitCellFilter(atoms)
# BFGS gives bad result, try fire?
opt = FIRE(ucf)
# fmax = 0.05 was kank
opt.run(fmax=0.01) 

# results = atoms.get_potential_energy()
results = atoms.calc.singlepoint
energies = results['orbital_energies']      # Shape: (nspin, nkpts, nbands)
orbitaloccupations = results['orbital_occupations']

occupied_indices = np.where(orbital_occupations > 1.8)[0]
homo_index = occupied_indices[-1]  
print("orbital occupancies: ", orbital_occupations[np.where(orbital_occupations > 0.5)])
print("homo occupation: ", orbital_occupations[homo_index])
print("homo energy: ", orbital_energies[homo_index])
unoccupied_indices = np.where(orbital_occupations < 0.1)[0]
lumo_index = unoccupied_indices[0]  
print("orbital unoccupancies: ", orbital_occupations[np.where(orbital_occupations <= 0.5)])
print("lumo occupation: ", orbital_occupations[lumo_index])
print("lumo energy: ", orbital_energies[lumo_index])
gap = (orbital_energies[lumo_index] - orbital_energies[homo_index])




# from ase.dft.bandgap import bandgap
# bg = bandgap(calc=atoms.calc)
# print("bg: ", bg)
# AttributeError: 'TBLiteASECalculator' object has no attribute 'get_number_of_spins'

# gap_ev = atoms.calc.results.get('bandgap', np.nan)
# if np.isfinite(gap_ev):
#     print(f"Bandgap (eV): {gap_ev}")
# else:
#     print("Bandgap not available from calculator results.")
# results:

# -------------------------------------------------------------------------- #


# # DOS

# hartree_to_ev = 27.2114
# energies_ev = orbital_energies * hartree_to_ev

# bin_width = 0.5  
# energy_min = energies_ev.min() - 1
# energy_max = 1
# bins = np.arange(energy_min, energy_max + bin_width, bin_width)
# weights = orbital_occupations

# plt.figure(figsize=(6, 4))
# plt.hist(energies_ev, bins=bins, weights=weights, color='royalblue', alpha=0.7, edgecolor='black')
# plt.xlabel('Energy (eV)')
# plt.ylabel('Density of States (arb. units)')
# plt.title('Density of States (DOS)')
# plt.tight_layout()
# plt.savefig('dos_plot.png')
# plt.close()
