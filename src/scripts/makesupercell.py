from ase.io import read, write

unit_cell = read('CdS.poscar', format='vasp')

supercell = unit_cell * (3, 3, 2)

write('POSCAR_3x3x2', supercell, format='vasp', vasp5=True, direct=True)
