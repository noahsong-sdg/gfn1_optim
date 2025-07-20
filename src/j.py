
from ase import Atoms, bulk
from ase import dft
from ase.calculators.xtb import XTB
from ase.calculators.calculator import Calculator
# https://wiki.fysik.dtu.dk/ase/ase/dft/bandgap.html


sys = bulk("CdS", "wurtzite", a=5.82, c=11.12)
def xtb_calc(tel=300):
    return TBLite(method="GFN1-xTB", electronic_temperature=tel)
calc = xtb_calc(tel=tel) 
ucf = UnitCellFilter(atoms)
opt = BFGS(ucf)


bandgap = dft.bandgap.bandgap(calc, direct, )

