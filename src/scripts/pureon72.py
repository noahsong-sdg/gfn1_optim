import qcelemental as qcel
import tblite.interface as tb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ase import Atoms
from ase.build import bulk
from tblite.ase import TBLite
from ase.optimize import BFGS
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
molecule = qcel.models.Molecule.from_file("cspbi3.xyz")
syms = molecule.symbols
geometry_angstroms = molecule.geometry * 0.529177210903

atoms = Atoms(symbols=syms, positions=geometry_angstroms)
atoms.set_cell([4.85, 10.65, 18.03])
atoms.set_pbc(True)
atoms.calc = TBLiteASECalculator(
    param_file="config/gfn1-base.toml",  # Path to your parameter file
    method="gfn1",
    electronic_temperature=300.0,
    charge=0.0,
    spin=0
)

ucf = UnitCellFilter(atoms)
opt = BFGS(ucf)
opt.run(fmax=0.05) 

results = atoms.get_potential_energy()


print("potential energy: ", results)
from ase.dft.bandgap import bandgap
# homo, lumo = atoms.get_homo_lumo_levels()
bg = bandgap(calc=atoms.calc)
print("bg: ", bg)
# AttributeError: 'TBLiteASECalculator' object has no attribute 'get_number_of_spins'
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
