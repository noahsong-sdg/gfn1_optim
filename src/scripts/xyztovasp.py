# holy shit ase has a function that speedruns this for me 

from ase.io import read, write
import os

structures = read('../trainall.xyz', index=':')

for i, atoms in enumerate(structures):
    namedir = f'vasp_structs/structure_{i:03d}'
    os.makedirs(namedir, exist_ok=True)
    write(f'{namedir}/POSCAR', atoms, format='vasp')
