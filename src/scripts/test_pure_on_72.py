import qcelemental as qcel
import tblite.interface as tb
import numpy as np
molecule = qcel.models.Molecule.from_file("pure_72_test.xyz")
xtb = tb.Calculator("GFN1-xTB", molecule.atomic_numbers, molecule.geometry)

results = xtb.singlepoint()
results["energy"]

orbital_energies = results["orbital-energies"]
orbital_occupations = results["orbital-occupations"]

occupied_indices = np.where(orbital_occupations > 0.75)[0]
homo_index = occupied_indices[-1]  
print(occupied_indices)
print("homo occupation: ", orbital_occupations[homo_index])
print("homo energy: ", orbital_energies[homo_index])
unoccupied_indices = np.where(orbital_occupations < 0.25)[0]
lumo_index = unoccupied_indices[0]  
print("lumo occupation: ", orbital_occupations[lumo_index])
print("lumo occupation: ", orbital_energies[lumo_index])
gap = (orbital_energies[lumo_index] - orbital_energies[homo_index])

print(results["energy"])
print(gap)


# lumo_index = np.argmax(orbital_occupations)
# homo_index = lumo_index - 1
# gap = (orbital_energies[lumo_index] - orbital_energies[homo_index])
# print(results["energy"] * 27.2)
# print(gap * 27.2)
