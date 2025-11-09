# DFTB+ Configuration Files

## HSD Template Files

These HSD (Human-readable Structured Data) files are templates for DFTB+ calculations using the xTB Hamiltonian.

### Available Templates

1. **`dftbp_xtb.hsd`** - Basic GFN1-xTB calculation
   - For molecular and periodic systems
   - K-points disabled by default
   - Use for quick calculations

2. **`dftbp_xtb_kpts.hsd`** - GFN1-xTB with k-point sampling
   - For periodic systems (crystals, surfaces)
   - 4×4×4 k-point grid by default
   - Includes Fermi smearing for metals

3. **`dftbp_gfn2.hsd`** - GFN2-xTB calculation
   - Better for conformational energies and reaction barriers
   - **NOT recommended for band gaps** (use GFN1-xTB instead)
   - Slightly slower

## Usage

### Direct Usage with DFTB+

```bash
# 1. Create structure file
cat > struc.xyz << EOF
3
Water molecule
O  0.0000  0.0000  0.0000
H  0.7580  0.5860  0.0000
H -0.7580  0.5860  0.0000
EOF

# 2. Copy template
cp config/dftbp_xtb.hsd dftb_in.hsd

# 3. Modify if needed (k-points, temperature, etc.)
# Edit dftb_in.hsd with your text editor

# 4. Run DFTB+
dftb+

# 5. Check results
cat detailed.out
cat band.out
```

### Modifying K-points

For periodic systems, edit the HSD file to set k-point grid:

```hsd
KPointsAndWeights = SupercellFolding {
  8 0 0    # nx divisions
  0 8 0    # ny divisions
  0 0 8    # nz divisions
  0.0 0.0 0.0  # shift (usually 0,0,0)
}
```

### Setting Electronic Temperature

For metallic systems, enable Fermi smearing:

```hsd
Filling = Fermi {
  Temperature [Kelvin] = 300.0
}
```

### Setting System Charge

For charged systems:

```hsd
Charge = 1.0  # +1 charged system
```

## Key HSD Sections

### Geometry

```hsd
Geometry = xyzFormat {
  <<< "struc.xyz"
}
```

The `<<<` operator includes the contents of `struc.xyz`. You can also use:
- `GenFormat` for DFTB+ gen format
- `VaspFormat` for VASP POSCAR format

### Hamiltonian

```hsd
Hamiltonian = xTB {
  Method = "GFN1-xTB"  # or "GFN2-xTB"
  SCC = Yes
  MaxSCCIterations = 100
}
```

### Options

```hsd
Options {
  WriteDetailedOut = Yes    # Main output file
  WriteResultsTag = Yes     # Machine-readable results
}
```

### Analysis

```hsd
Analysis {
  WriteBandOut = Yes        # Eigenvalues and occupations (band.out file)
  CalculateForces = Yes     # Compute atomic forces
}
```

## Output Files

After running DFTB+, the following files are created:

- **`detailed.out`**: Main output file
  - Total energy
  - Forces on atoms
  - Mulliken charges
  - Fermi level
  - Convergence info

- **`band.out`**: Band structure data
  - Eigenvalues (eV)
  - Orbital occupations
  - For each k-point

- **`results.tag`**: Machine-readable results
  - Easy to parse programmatically

## Advantages of xTB Hamiltonian

Using `Hamiltonian = xTB` instead of `Hamiltonian = DFTB`:

1. **No SKF files needed**: xTB parameters are built into DFTB+
2. **Covers all elements**: Up to element 86 (Rn)
3. **Well-tested**: GFN1-xTB and GFN2-xTB are widely used
4. **Consistent**: Same parameters across all systems
5. **Fast**: Optimized implementation

### GFN1-xTB vs GFN2-xTB

**For band gap calculations: Use GFN1-xTB**

- **GFN1-xTB**: Better for band gaps, excitation energies, periodic systems
- **GFN2-xTB**: Better for conformational energies, reaction barriers, molecular properties

## Tips

### For Molecules (Non-periodic)

```hsd
# No k-points needed
# Charge and spin can be set
Charge = 0.0
```

### For Metals

```hsd
# Use electronic temperature for SCF stability
Filling = Fermi {
  Temperature [Kelvin] = 1000.0  # Higher for difficult cases
}

# Lower mixing parameter
Mixer = Broyden {
  MixingParameter = 0.05
}
```

### For Large Systems

```hsd
# Enable parallel execution
Parallel {
  UseOmpThreads = Yes
}

# Set number of threads via environment variable:
# export OMP_NUM_THREADS=8
```

### For Band Gap Calculations

**Important**: Use GFN1-xTB, NOT GFN2-xTB, for band gap calculations. GFN1-xTB has been shown to predict band gaps more accurately for semiconductors and insulators.

```hsd
Hamiltonian = xTB {
  Method = "GFN1-xTB"  # IMPORTANT: Use GFN1-xTB for band gaps!
}

# Must have these options
Options {
  WriteDetailedOut = Yes
}

Analysis {
  WriteBandOut = Yes
}

# Use sufficient k-points for convergence
KPointsAndWeights = SupercellFolding {
  6 0 0
  0 6 0
  0 0 6
  0.0 0.0 0.0
}
```

## Example: Band Gap of Si

```bash
# 1. Create Si structure
cat > struc.xyz << EOF
2
Si diamond structure (primitive cell)
Si  0.0000  0.0000  0.0000
Si  1.3575  1.3575  1.3575
EOF

# 2. Use k-point template
cp config/dftbp_xtb_kpts.hsd dftb_in.hsd

# 3. Edit to increase k-points
# Change to 8 8 8 grid

# 4. Run DFTB+
dftb+

# 5. Extract band gap from band.out
# Look for gap between highest occupied and lowest unoccupied states
```

## References

- [DFTB+ Manual](https://dftbplus.org/documentation)
- [DFTB+ Recipes](https://dftbplus-recipes.readthedocs.io/)
- [xTB Documentation](https://xtb-docs.readthedocs.io/)

