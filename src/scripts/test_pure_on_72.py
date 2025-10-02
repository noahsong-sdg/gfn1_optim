import qcelemental as qcel
import tblite.interface as tb
import numpy as np
molecule = qcel.models.Molecule.from_file("pure_72_test.xyz")

xtb = tb.Calculator("GFN1-xTB", molecule.atomic_numbers, molecule.geometry)

results = xtb.singlepoint()
results["energy"]
orbital_energies = results["orbital-energies"]
orbital_occupations = results["orbital-occupations"]

lumo_index = np.argmax(orbital_occupations)
homo_index = lumo_index - 1
gap = (orbital_energies[lumo_index] - orbital_energies[homo_index]) * qcel.constants.conversion_factor("hartree", "eV")
print(results["energy"])
print(gap)
