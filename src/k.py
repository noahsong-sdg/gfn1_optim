"""
Test script for TBLite ASE calculator integration
"""

import numpy as np
from ase.build import bulk 
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from tblite_ase_calculator import TBLiteASECalculator

if __name__ == "__main__":
    volumes = []
    a_params = []
    c_params = []
    u_params = []

    print("\nTesting solid state optimization...")
    cds_def = bulk("CdS", "wurtzite", a=4.17, c=6.78)
    cds_def.calc = TBLiteASECalculator("config/gfn1-base.toml", method="gfn1")
    ucf = UnitCellFilter(cds_def)
    opt = BFGS(ucf)
    opt.run(fmax=0.01) 
    cellpars = cds_def.cell.cellpar()
    a_params.append(cellpars[0])
    c_params.append(cellpars[2])
    print("default:",a_params, c_params)

    cds = bulk("CdS", "wurtzite", a=4.17, c=6.78)
    cds.calc = TBLiteASECalculator("results/parameters/CdS_optimized_bayesian.toml", method="gfn1")
    ucf = UnitCellFilter(cds)
    opt = BFGS(ucf)
    opt.run(fmax=0.01) 
    cellpars = cds.cell.cellpar()
    a = cellpars[0]
    c = cellpars[2]
    print("optimized:",a, c)
    
    
