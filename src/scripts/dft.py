import numpy as np
import pandas as pd
import ase.io
import os
import time
from pathlib import Path
from gpaw import GPAW, PW, FermiDirac, mpi
from gpaw.poisson import PoissonSolver
import ase
from ase.build import bulk
from tblite.ase import TBLite
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
import matplotlib.pyplot as plt
from ase import Atoms

atoms = bulk("CdS", "wurtzite", a = 4.17, c = 6.78)

# Step 1: Relax cell + positions with a GGA (supports stress)
calc = GPAW(
        mode=PW(400),
        xc='PBE',
        kpts=(2, 2, 2),
        occupations=FermiDirac(0.01),
        txt=None
    )

atoms.calc = calc
ucf = UnitCellFilter(atoms)
opt = BFGS(ucf)
opt.run(fmax=0.02)

# Step 2: Single-point with hybrid (Gamma-only; no stress)
calc_pbe0 = GPAW(
        mode=PW(400),
        xc='PBE0',
        kpts=(1, 1, 1),  # Gamma-only for hybrids
        occupations=FermiDirac(0.01),
        poissonsolver=PoissonSolver(use_charge_center=True),
        txt=None
    )
atoms.calc = calc_pbe0
energy = atoms.get_potential_energy() * 0.0367493  # eV -> Hartree
df = pd.DataFrame({
        'a': [atoms.cell.cellpar()[0]],
        'b': [atoms.cell.cellpar()[1]],
        'c': [atoms.cell.cellpar()[2]],
        'alpha': [atoms.cell.cellpar()[3]],
        'beta': [atoms.cell.cellpar()[4]],
        'gamma': [atoms.cell.cellpar()[5]],
        'Energy': [energy],  # Convert to Hartree
        #'bandgap': self.get_bandgap(),
        #'elastic': self.getElastic()
    })

print(df)

#           a         b         c  alpha  beta  gamma    Energy
# 0  3.892551  3.892551  6.628439   90.0  90.0  120.0 -0.382083
