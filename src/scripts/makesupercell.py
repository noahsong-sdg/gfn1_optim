from ase.io import read, write, vasp

# cell = vasp.read_vasp("CdS.poscar")
# supercell = cell * (3, 3, 2)
# vasp.write_vasp('POSCAR', supercell, vasp5=True, direct=True)

import ase.io
from ase.build import make_supercell
unit_cell = ase.io.read('CdS.poscar')
supercell = unit_cell.repeat((3, 3, 2))
ase.io.vasp.write_vasp('POSCAR_sort', supercell, direct=True, sort=True)
# ase.io.write('POSCAR_supercell', supercell, format='vasp')

# conversion
# atoms = read('POSCAR_supercell')
# write('pure_72_test.xyz', atoms)


# transformation_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 2]]
# supercell = make_supercell(unit_cell, transformation_matrix)
