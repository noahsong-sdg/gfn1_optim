"""
pure: {'a': 4.812134508597587, 'c': 7.458952249253911, 'alpha': 89.99999999999994, 'beta': 89.99999999999996, 
'gamma': 119.99999972975012, 'volume': 149.58359831772657, 'u': 0.4630666817140237}

{'a': 4.812134508597609, 'c': 7.458952249253893, 'alpha': 90.0, 'beta': 89.99999999999996, 
'gamma': 119.99999972975007, 'volume': 149.58359831772762, 'u': 0.46306668171402454}


"""



from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from ase.build import bulk 
from tblite.ase import TBLite
from ase.dft import bandgap

from typing import Optional, Union, List, Tuple
from enum import Enum
from dataclasses import dataclass

from config import SystemConfig, SystemType, CalculationType, get_system_config
from config import get_calculation_distances, create_molecule_geometry, get_isolated_atom_symbol

# https://wiki.fysik.dtu.dk/ase/ase/dft/bandgap.html
class CalcMethod(Enum):
    DFT = "dft"
    XTB = "xtb"
    CUSTOM = "custom"

@dataclass 
class CalcConfig:
    method: CalcMethod 
    basis: str = "cc-pVTZ"
    elec_temp: float = 300.0
    param_file: Optional[str] = None
    spin: int = 0
    custom_param_file: Optional[str] = None

class Calc():
    def __init__(self, calc_config: CalcConfig, system_config: SystemConfig):
        self.calc_config = calc_config
        self.system_config = system_config
    
    def calc_energy(self, atoms) -> float:
        """Calculate energy for given atomic configuration"""
        # Suppress verbose output during optimization
        if self.calc_config.method == CalcMethod.DFT:
            return self._calc_dft(atoms)
        elif self.calc_config.method == CalcMethod.XTB:
            return self._calc_xtb(atoms)
        elif self.calc_config.method == CalcMethod.CUSTOM:
            return self._calc_custom_tblite(atoms)
        else:
            raise ValueError(f"Unsupported method: {self.calc_config.method}")

def xtb_calc(tel=300):
    return TBLite(method="GFN1-xTB", electronic_temperature=tel)
TEL = 300
##############################################################################

atoms = bulk("CdS", "wurtzite", a=4.17, c=6.78)
atoms.calc = TBLite(method="GFN1-xTB", param="results/parameters/test.toml", electronic_temperature=TEL) 
# atoms.calc = TBLite(method="GFN1-xTB", electronic_temperature=TEL)
ucf = UnitCellFilter(atoms)
opt = BFGS(ucf)
opt.run(fmax=0.01) # Convergence criterion: max force < 0.01 eV/Å

cellpars = atoms.cell.cellpar()
scaled_positions = atoms.get_scaled_positions()
u_val = scaled_positions[2, 2] 
# Store results

# gap = dft.bandgap.bandgap(calc, )
properties_gfn = {
        'a': cellpars[0],  # Length of the first lattice vector (a)
        'c': cellpars[2],  # Length of the third lattice vector (c)
        'alpha': cellpars[3],  # Angle between b and c (alpha)
        'beta': cellpars[4],  # Angle between a and c (beta)
        'gamma': cellpars[5],  # Angle between a and b (gamma)
        'volume': atoms.get_volume(),  # Volume of the unit cell (Angstrom^3)
        'u': u_val,
        # 'Band Gap': gap,
        # 'elastic_constants_GPa': elastic, # 6x6 Voigt matrix
    }
print(properties_gfn)



properties_custom = {
    'a': cellpars[0],  # Length of the first lattice vector (a)
    'c': cellpars[2],  # Length of the third lattice vector (c)
    'alpha': cellpars[3],  # Angle between b and c (alpha)
    'beta': cellpars[4],  # Angle between a and c (beta)
    'gamma': cellpars[5],  # Angle between a and b (gamma)
    'volume': atoms.get_volume(),  # Volume of the unit cell (Angstrom^3)
    'u': u_val,
}



