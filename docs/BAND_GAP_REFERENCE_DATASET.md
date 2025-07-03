# Band Gap Reference Dataset for TBLite Parameter Optimization

## Overview

This document describes the approach for creating a high-quality reference dataset of band gaps for CdS structures to be used in TBLite parameter optimization.

## Problem Statement

You need to compute band gaps at the gamma point for each structure in your CdS dataset (`trainall.xyz` and `val_lind50_eq.xyz`) using a higher level of theory than TBLite to create a reference dataset for parameter optimization.

## Solution: HPC-Optimized PySCF Calculator

### Why PySCF for HPC?

1. **Available in Environment**: PySCF is already installed in your pixi environment
2. **HPC Scalability**: With proper optimization, PySCF can handle 72-atom systems on HPC
3. **Quality**: Provides DFT-level accuracy suitable for reference data
4. **Flexibility**: Easy to modify settings for different accuracy/speed trade-offs

### Key Optimizations for 72-Atom Systems

#### Memory Management
- **Max Memory**: 8 GB limit per calculation
- **Coarse Mesh**: 25×25×25 grid for faster calculations
- **Relaxed Precision**: 1e-8 tolerance for speed

#### SCF Convergence
- **Mixing Parameter**: 0.7 for better convergence
- **Level Shift**: 0.2 Hartree to avoid level crossing
- **Davidson Diagonalization**: Efficient for large systems
- **Max Cycles**: 100 (reduced from default 350)

#### Basis Set
- **GTH-DZVP**: Double-zeta basis with pseudopotentials
- **Balanced**: Good accuracy without excessive computational cost

### Implementation

#### Files Created

1. **`src/hpc_band_gap_calculator.py`**: Main calculator class
2. **`test_hpc_band_gap.py`**: Test script for verification
3. **`slurm/run_band_gap_calc.sh`**: SLURM job script for HPC

#### Key Features

- **Gamma Point Only**: Focuses on Γ-point band gaps as requested
- **Proper Electron Counting**: Correctly handles Cd (48 electrons) and S (16 electrons)
- **Error Handling**: Robust error handling for failed calculations
- **Progress Tracking**: Shows progress for large datasets
- **Comprehensive Output**: Band gap, VBM, CBM, Fermi energy, total energy

### Usage

#### Local Testing
```bash
# Test with single structure
python test_hpc_band_gap.py

# Run full calculation
python src/hpc_band_gap_calculator.py
```

#### HPC Submission
```bash
# Submit to SLURM
sbatch slurm/run_band_gap_calc.sh

# Check status
squeue -u $USER
```

### Expected Performance

#### Computational Requirements
- **Memory**: ~8 GB per calculation
- **CPU**: 8 cores recommended
- **Time**: ~30-60 minutes per structure (72 atoms)
- **Total**: ~24 hours for full dataset (200+ structures)

#### Output Files
- `results/train_band_gaps_hpc.csv`: Training set band gaps
- `results/val_band_gaps_hpc.csv`: Validation set band gaps
- `results/*_band_gap_dist_hpc.png`: Distribution plots

### Alternative Approaches Considered

#### 1. Quantum ESPRESSO
- **Pros**: Industry standard, excellent scalability
- **Cons**: Not available in current environment, requires installation
- **Status**: Code prepared but not implemented

#### 2. TBLite (Rejected)
- **Pros**: Fast, already available
- **Cons**: Same level as optimization target - not suitable for reference
- **Status**: Implemented but not used for reference data

#### 3. PySCF Standard
- **Pros**: Available, accurate
- **Cons**: Poor scalability for 72 atoms without optimization
- **Status**: Enhanced with HPC optimizations

### Quality Assurance

#### Validation Steps
1. **Single Structure Test**: Verify calculation works with one structure
2. **Multiple Structure Test**: Test with 3-5 structures
3. **Full Dataset**: Process complete dataset
4. **Statistical Analysis**: Check band gap distributions
5. **Comparison**: Compare with literature values for CdS

#### Expected Results
- **Band Gap Range**: 1.5-3.5 eV (typical for CdS)
- **Mean Band Gap**: ~2.4 eV (bulk CdS)
- **Success Rate**: >90% for well-converged structures

### Integration with Optimization

#### Reference Dataset Format
```csv
structure_id,n_atoms,cell_volume,chemical_formula,n_cd,n_s,band_gap,vbm,cbm,fermi_energy,total_energy,is_direct,error
0,72,1234.56,Cd36S36,36,36,2.456,-1.234,1.222,0.0,-12345.67,True,
```

#### Optimization Workflow
1. **Generate Reference**: Run band gap calculations
2. **Train TBLite**: Optimize parameters against reference
3. **Validate**: Test on validation set
4. **Iterate**: Refine parameters as needed

### Troubleshooting

#### Common Issues
1. **Memory Errors**: Reduce `max_memory` or use fewer cores
2. **Convergence Failures**: Increase `max_cycle` or adjust mixing
3. **Timeout**: Increase `max_seconds` in SLURM script
4. **Import Errors**: Check `PYTHONPATH` and module loading

#### Performance Tuning
- **Faster**: Reduce basis set, increase convergence tolerance
- **More Accurate**: Increase basis set, tighten convergence
- **More Stable**: Adjust mixing parameters, level shift

### Next Steps

1. **Test Locally**: Run `test_hpc_band_gap.py` to verify setup
2. **Submit to HPC**: Use SLURM script for full calculation
3. **Monitor Progress**: Check output files and logs
4. **Analyze Results**: Review band gap distributions
5. **Optimize TBLite**: Use reference data for parameter tuning

This approach provides a robust, scalable solution for generating high-quality reference band gap data suitable for TBLite parameter optimization on HPC systems. 
