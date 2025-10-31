# DFTB+ Band Gap Calculation

## Overview

This document describes the **DFTB+** implementation for band gap calculations. **DFTB+** is a standalone executable that provides full k-point support for accurate band structure calculations in periodic systems.

**Important**: This is **DFTB+** (the standalone program), not to be confused with **dxtb** (a PyTorch library). DFTB+ has mature support for k-point sampling, making it ideal for periodic solid-state calculations.

## Files Created

### 1. `src/calculators/dftbp_ase_calculator.py`

An ASE (Atomic Simulation Environment) calculator interface for DFTB+. This calculator:

- Wraps the DFTB+ executable as a subprocess
- Generates HSD (Human-readable Structured Data) input files
- Parses DFTB+ output files (detailed.out, band.out)
- Supports full k-point grids for band structure calculations
- Handles both periodic and non-periodic systems
- Calculates: energy, forces, charges, and band gaps

**Key Features:**
- **Full K-point Support**: Unlike dxtb, DFTB+ supports arbitrary k-point grids
- **Production-Ready**: DFTB+ is a mature, well-tested code
- **Slater-Koster Parameters**: Uses standard SK parameter sets (mio, 3ob, matsci, etc.)
- **SCC Convergence**: Self-consistent charge calculations with multiple mixer options

### 2. `src/scripts/dftbp.py`

A script to compute band gaps using the DFTB+ calculator. This script:

1. Loads a crystal structure (CsPbI3 by default)
2. Performs geometry optimization with coarse k-point grid
3. Calculates band structure with fine k-point grid
4. Extracts band gap from eigenvalues and occupations
5. Generates DOS visualization
6. Saves band structure data

## Installation

### Prerequisites

1. **DFTB+ executable**: Install DFTB+ on your system
   
   ```bash
   # Option 1: Using conda (recommended)
   conda install -c conda-forge dftbplus
   
   # Option 2: Build from source
   # See https://dftbplus.org/download for instructions
   ```

2. **Slater-Koster parameter files**: Download appropriate SK files

   ```bash
   # Download from https://dftb.org/parameters/download
   # Common parameter sets:
   # - mio-1-1: Organic molecules (C, H, N, O, S)
   # - 3ob-3-1: Extended set with more elements
   # - matsci-0-3: Materials science (metals, semiconductors)
   
   # Extract to a directory, e.g., ./slakos/
   ```

3. **Python dependencies**:

   ```bash
   pip install ase numpy matplotlib
   ```

### Setup

Set the environment variable for SK files:

```bash
export DFTB_SKF_DIR=/path/to/your/slater-koster-files
```

For example:

```bash
export DFTB_SKF_DIR=$HOME/slakos/mio-1-1
```

## Usage

### Basic Usage

Run the band gap calculation:

```bash
cd /home/noahsong/work/gfn/gfn1_optim
export DFTB_SKF_DIR=/path/to/sk/files
python src/scripts/dftbp.py
```

This will:
- Read `cspbi3.cif` from the current directory
- Optimize the structure using k-point grid (2,2,2)
- Calculate band structure using k-point grid (4,4,4)
- Save relaxed structure to `relaxed_cspbi3_dftbp.{cif,xyz}`
- Generate DOS plot `dos_dftbp.png`
- Save band data to `band_data_dftbp.txt`

### Using the Calculator Directly

```python
from ase.io import read
from calculators.dftbp_ase_calculator import DFTBPlusASECalculator

# Load structure
atoms = read('structure.cif')

# Define maximum angular momentum for each element
# This depends on your SK parameter set
max_angular = {
    'H': 's',   # H has s orbitals only
    'C': 'p',   # C has s, p orbitals
    'N': 'p',   # N has s, p orbitals
    'O': 'p',   # O has s, p orbitals
}

# Create calculator
calc = DFTBPlusASECalculator(
    skf_dir='/path/to/sk/files',
    max_angular=max_angular,
    kpts=(4, 4, 4),              # K-point grid
    scc=True,                     # Self-consistent charges
    max_scc_iterations=100,
    electronic_temperature=300.0, # in Kelvin
    write_band_out=True           # Enable band structure output
)

# Attach calculator to atoms
atoms.calc = calc

# Get properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()             # eV/Angstrom

# Access band gap
if 'bandgap' in calc.results:
    print(f"Band gap: {calc.results['bandgap']:.4f} eV")

# Access eigenvalues
eigenvalues = calc.get_eigenvalues()
occupations = calc.get_occupations()
```

### Geometry Optimization

```python
from ase.optimize import FIRE
from ase.filters import UnitCellFilter

# Set up calculator
atoms.calc = DFTBPlusASECalculator(
    skf_dir='/path/to/sk/files',
    max_angular=max_angular,
    kpts=(2, 2, 2)  # Coarse grid for optimization
)

# Optimize cell and atomic positions
ucf = UnitCellFilter(atoms)
opt = FIRE(ucf)
opt.run(fmax=0.01)
```

## Configuration

### Slater-Koster Parameter Sets

Different parameter sets for different systems:

| Parameter Set | Elements | Use Cases |
|--------------|----------|-----------|
| mio-1-1 | H, C, N, O, S | Organic molecules, biomolecules |
| 3ob-3-1 | H, C, N, O, P, S, Na, Mg, Zn, F, Cl, Br, I, Ca | Extended organic chemistry |
| matsci-0-3 | Many metals and semiconductors | Solid-state materials, alloys |
| pbc-0-3 | Periodic boundary conditions optimized | Bulk materials |

### Maximum Angular Momentum

Check your SK parameter set documentation for the maximum angular momentum:

```python
# Example for mio-1-1 set
max_angular = {
    'H': 's',   # 1 shell: s
    'C': 'p',   # 2 shells: s, p
    'N': 'p',   # 2 shells: s, p
    'O': 'p',   # 2 shells: s, p
    'S': 'd',   # 3 shells: s, p, d
}

# Example for metals (matsci set)
max_angular = {
    'Fe': 'd',  # s, p, d orbitals
    'Cu': 'd',  # s, p, d orbitals
    'Ti': 'd',  # s, p, d orbitals
}
```

### K-point Grids

Choose appropriate k-point grids based on your system:

```python
# Molecules (non-periodic)
kpts = (1, 1, 1)

# Small unit cell bulk materials
kpts = (8, 8, 8)

# Medium unit cell
kpts = (4, 4, 4)

# Large supercell
kpts = (2, 2, 2)

# 2D materials (periodic in xy, non-periodic in z)
kpts = (8, 8, 1)

# 1D materials (periodic along x only)
kpts = (12, 1, 1)
```

## Key Differences: DFTB+ vs TBLite

| Feature | TBLite | DFTB+ |
|---------|--------|-------|
| Implementation | Command-line tool | Standalone executable |
| Backend | Fortran | Fortran |
| K-point Support | Limited | Full support |
| Parameter Format | TOML (custom) | SKF files (standard) |
| Community | Small | Large, established |
| Documentation | Limited | Extensive |
| Maturity | New | Mature (15+ years) |
| Band Structure | Basic | Advanced |
| Features | Basic tight-binding | Extended features (TD-DFTB, REKS, etc.) |

## API Reference

### DFTBPlusASECalculator

#### Constructor Parameters

- `skf_dir` (str, required): Directory containing Slater-Koster files (.skf)
  
- `max_angular` (dict, required): Maximum angular momentum for each element
  - Example: `{'H': 's', 'C': 'p', 'O': 'p'}`
  - Values: `'s'`, `'p'`, `'d'`, `'f'`

- `kpts` (tuple): K-point grid (nx, ny, nz)
  - Default: `(1, 1, 1)`
  - Example: `(4, 4, 4)` for 4×4×4 Monkhorst-Pack grid

- `scc` (bool): Enable self-consistent charge calculation
  - Default: `True`

- `max_scc_iterations` (int): Maximum SCC iterations
  - Default: `100`

- `electronic_temperature` (float): Electronic temperature in Kelvin
  - Default: `0.0` (T=0K)
  - For metals, use ~300-1000K for better convergence

- `charge` (float): Total system charge
  - Default: `0.0`

- `mixer_param` (float): Mixing parameter for SCC convergence
  - Default: `0.2`
  - Range: 0.01-0.5 (lower = more stable, slower)

- `write_band_out` (bool): Write band.out file with eigenvalues
  - Default: `True`

- `dftb_command` (str): Command to run DFTB+
  - Default: `'dftb+'`
  - Use custom path if needed: `'/path/to/dftb+'`

#### Methods

- `calculate(atoms, properties, system_changes)`: Calculate properties
- `get_eigenvalues(kpt, spin)`: Get orbital eigenvalues (eV)
- `get_occupations(kpt, spin)`: Get orbital occupations
- `get_fermi_level()`: Get Fermi level (eV)
- `get_number_of_spins()`: Get number of spin channels
- `get_ibz_k_points()`: Get irreducible k-points

#### Results Dictionary

After calculation, the following properties are available in `calc.results`:

- `energy`: Total energy (eV)
- `forces`: Atomic forces (eV/Angstrom)
- `charges`: Mulliken charges
- `bandgap`: HOMO-LUMO gap (eV)

## Debugging

Enable verbose output:

```bash
export DFTBP_PRINT_STDOUT=1
python src/scripts/dftbp.py
```

This will print DFTB+ stdout and stderr to the console.

## Troubleshooting

### Error: "dftb+ command not found"

**Solution**: Install DFTB+ or set full path:

```python
calc = DFTBPlusASECalculator(
    dftb_command='/full/path/to/dftb+',
    ...
)
```

### Error: "SKF file not found"

**Solution**: Check your SKF directory and ensure all element pairs exist:

```bash
# For elements H and O, you need:
# H-H.skf, H-O.skf, O-H.skf, O-O.skf
ls $DFTB_SKF_DIR/*.skf
```

### SCC Convergence Issues

**Solution 1**: Reduce mixing parameter:

```python
calc = DFTBPlusASECalculator(
    mixer_param=0.05,  # Lower = more stable
    max_scc_iterations=200
)
```

**Solution 2**: Increase electronic temperature:

```python
calc = DFTBPlusASECalculator(
    electronic_temperature=1000.0  # Higher temp for metals
)
```

### Band Gap is Zero (Metal)

This may be correct for metallic systems. For semiconductors:

1. **Check convergence**: Ensure SCC converged
2. **Increase k-points**: Use finer grid, e.g., (8,8,8)
3. **Check parameters**: Verify correct SK parameter set

### Forces are Wrong

**Solution**: Check unit conversion in the code. DFTB+ outputs forces in Hartree/Bohr, which are converted to eV/Angstrom by factor of 51.4220652.

## Example Output

```
======================================================================
Band Gap Calculation using DFTB+
Full k-point sampling for accurate band structure
======================================================================

[Step 1] Loading structure and setting up optimization...
  Loaded structure: 20 atoms
  Formula: CsPbI3
  Initial cell volume: 248.56 Angstrom^3
  Periodic: [ True  True  True]

[Step 2] Optimizing geometry with FIRE algorithm...
  Using k-point grid: (2, 2, 2)
  SKF directory: ./slakos/mio-ext
  Optimization converged after 52 steps
  Final cell volume: 245.87 Angstrom^3
  Final energy: -12456.7891 eV
  Relaxed structure saved to: relaxed_cspbi3_dftbp.{cif,xyz}

[Step 3] Computing band structure with fine k-point grid...
  Using k-point grid: (4, 4, 4)
  Total energy: -12456.7891 eV

[Step 4] Extracting band gap...
  Number of eigenvalues: 320
  Eigenvalue range: -18.56 to 15.42 eV

======================================================================
RESULTS
======================================================================
Valence Band Maximum (VBM):     -4.5234 eV
Conduction Band Minimum (CBM):  -2.8567 eV
Band Gap:                       1.6667 eV
Fermi Level:                    -3.6900 eV
Number of occupied levels:      160
Number of unoccupied levels:    160
======================================================================

  DOS plot saved to: dos_dftbp.png
  Band data saved to: band_data_dftbp.txt
```

## Advanced Features

### Custom Input Files

You can modify the HSD input generation in `_write_dftb_input()` to add advanced features:

```python
# In dftbp_ase_calculator.py, modify _write_dftb_input to add:

# 1. Dispersion corrections
f.write('  Dispersion = DftD3 {\n')
f.write('    Damping = BeckeJohnson {\n')
f.write('      a1 = 0.5719\n')
f.write('      a2 = 3.6017\n')
f.write('    }\n')
f.write('  }\n')

# 2. Spin polarization
f.write('  SpinPolarisation = Colinear {\n')
f.write('    UnpairedElectrons = 2\n')
f.write('  }\n')

# 3. External electric field
f.write('  ElectricField {\n')
f.write('    External {\n')
f.write('      Direction = 0.0 0.0 1.0\n')
f.write('      Strength [V/A] = 0.1\n')
f.write('    }\n')
f.write('  }\n')
```

### Band Structure Along Specific Paths

For band structure plots along high-symmetry paths, see the DFTB+ documentation on band structure calculations.

## References

- [DFTB+ Official Website](https://dftbplus.org/)
- [DFTB+ Recipes (Documentation)](https://dftbplus-recipes.readthedocs.io/)
- [DFTB+ GitHub Repository](https://github.com/dftbplus/dftbplus)
- [Slater-Koster Parameters](https://dftb.org/parameters)
- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)

## Citation

If you use DFTB+ in your research, please cite:

```
B. Hourahine, B. Aradi, V. Blum, F. Bonafé, A. Buccheri, C. Camacho,
C. Cevallos, M. Y. Deshaye, T. Dumitrică, A. Dominguez, S. Ehlert,
M. Elstner, T. van der Heide, J. Hermann, S. Irle, J. J. Kranz,
C. Köhler, T. Kowalczyk, T. Kubař, I. S. Lee, V. Lutsker, R. J. Maurer,
S. K. Min, I. Mitchell, C. Negre, T. A. Niehaus, A. M. N. Niklasson,
A. J. Page, A. Pecchia, G. Penazzi, M. P. Persson, J. Řezáč, C. G. Sánchez,
M. Sternberg, M. Stöhr, F. Stuckenberg, A. Tkatchenko, V. W.-z. Yu,
and T. Frauenheim, DFTB+, a software package for efficient approximate
density functional theory based atomistic simulations, J. Chem. Phys.
152, 124101 (2020); doi: 10.1063/1.5143190
```

## Notes

1. **K-point Convergence**: Always test k-point convergence for your specific system. Start with coarse grids and refine until band gap converges.

2. **Parameter Selection**: Choose appropriate SK parameter sets for your system. Mixing parameters from different sets is not recommended.

3. **Geometry Optimization**: Use coarser k-point grids during optimization to save time, then use fine grids for final properties.

4. **Metallic Systems**: For metals, use electronic temperature (Fermi-Dirac smearing) to aid SCF convergence.

5. **Comparison**: DFTB+ results should be validated against higher-level methods (DFT, experiment) for your specific system.

