# Codebase Guide: TBLite Parameter Optimization

## Project Overview

This project optimizes TBLite (GFN1-xTB) parameters for quantum chemistry calculations using various optimization algorithms (Genetic Algorithm, Particle Swarm Optimization, Bayesian Optimization). It supports both molecular systems (dissociation curves) and solid-state systems (lattice constants).

## Directory Structure

```
src/
├── optimizers/          # Optimization algorithms
│   ├── ga.py           # Custom Genetic Algorithm
│   ├── gad.py          # PyGAD-based Genetic Algorithm
│   ├── pso.py          # Particle Swarm Optimization
│   ├── cma.py          # CMA-ES optimization
│   └── bayes_h.py      # Bayesian optimization
├── calculators/         # Quantum chemistry calculators
│   ├── calc.py         # Main calculator interface
│   └── tblite_ase_calculator.py  # TBLite ASE wrapper
├── utils/              # Utilities and helpers
│   ├── parameter_bounds.py    # Parameter constraint management
│   ├── parameter_stats.py     # Parameter statistics and histograms
│   └── data_extraction.py     # Data processing utilities
├── tests/              # Test scripts and analysis
│   ├── lattice_results.py     # Lattice constant analysis
│   └── bandgap.py             # Bandgap calculations
├── config.py           # System configurations
├── base_optimizer.py   # Base class for all optimizers
├── cli.py              # Command-line interface
└── common.py           # Common constants and utilities

config/
├── gfn1-base.toml      # Default TBLite parameters
├── tbfit.toml          # Alternative parameter set
└── si_tbfit.toml       # Silicon-specific parameters

results/
├── parameters/         # Optimized parameter files
├── fitness/           # Fitness history data
├── curves/            # Dissociation curves
└── plots/             # Generated plots and histograms
```

## Key Components

### 1. Base Optimizer (`base_optimizer.py`)
- **Purpose**: Abstract base class for all optimization algorithms
- **Key Methods**: 
  - `evaluate_fitness()` - Evaluates parameter sets
  - `apply_bounds()` - Enforces parameter constraints
  - `create_param_file()` - Generates TOML parameter files
- **Inheritance**: All optimizers (GA, PSO, Bayesian) inherit from this

### 2. Parameter Bounds Management (`utils/parameter_bounds.py`)
- **Purpose**: Centralized parameter constraint system
- **Key Classes**:
  - `ParameterBoundsManager` - Manages scientific constraints
  - `ParameterBounds` - Individual parameter bounds
  - `create_10p_parameter_bounds()` - Creates ±10% bounds
- **Important**: Excludes `zeff`, `refocc`, `dkernel`, `qkernel`, `mprad`, `mpvcn` from optimization
- **Current Implementation**: Uses `create_10p_parameter_bounds()` by default (10% margins around defaults)
- **Static Bounds**: Available via `create_static_parameter_bounds()` using `PARAMETER_CONSTRAINTS`
- **Dynamic Bounds**: Available via `create_parameter_bounds()` (deprecated, now uses static)

### 3. Calculator Interface (`calculators/calc.py`)
- **Purpose**: Unified interface for quantum chemistry calculations
- **Supported Methods**: GFN1-xTB, CCSD, custom TBLite
- **Key Classes**:
  - `GeneralCalculator` - Main calculator interface
  - `DissociationCurveGenerator` - Molecular calculations
  - `CrystalGenerator` - Solid-state calculations

### 4. System Configuration (`config.py`)
- **Purpose**: Pre-defined system configurations
- **Key Systems**: H2, Si2, CdS
- **Configuration**: Bond ranges, lattice parameters, spin multiplicity

## Common Workflows

### Adding a New System
1. Edit `src/config.py` - Add to `SYSTEM_CONFIGS`
2. Define system type, elements, and target properties
3. Add reference data if needed

### Adding a New Optimizer
1. Inherit from `BaseOptimizer`
2. Implement `optimize()` method
3. Add to CLI options in `src/cli.py`

### Modifying Parameter Bounds
1. Edit `src/utils/parameter_bounds.py` - `PARAMETER_CONSTRAINTS`
2. Use `create_10p_parameter_bounds()` for ±10% margins
3. Validate with `ParameterBoundsManager.validate_parameters()`

### Running Optimizations
```bash
# From project root
PYTHONPATH=src python src/cli.py optimize --system H2 --method ga
PYTHONPATH=src python src/cli.py optimize --system CdS --method pso
PYTHONPATH=src python src/cli.py optimize --system Si2 --method bayes
```

## Important Design Decisions

### Parameter Selection
- **Excluded Parameters**: `zeff`, `refocc`, `dkernel`, `qkernel`, `mprad`, `mpvcn`
- **Reason**: These are fundamental constants, not tunable parameters
- **Included Parameters**: `levels`, `slater`, `shpoly`, `kcn`, `gam`, `lgam`, `gam3`, `arep`, `xbond`, `en`

### Fitness Functions
- **Molecular Systems**: RMSE between calculated and reference dissociation curves
- **Solid-State Systems**: Squared error in lattice constants (a, b, c) - see `base_optimizer.py` line 189-200
- **Energy Units**: Hartree (converted from eV in calculators)
- **Lattice Fitness**: `loss = (a_opt - a_ref)² + (b_opt - b_ref)² + (c_opt - c_ref)²`

### Bounds Management
- **Current Default**: Uses `create_10p_parameter_bounds()` with ±10% margins around default values
- **Static Bounds**: Based on physical constraints in `PARAMETER_CONSTRAINTS`
- **Dynamic Bounds**: ±10% of default values using `create_10p_parameter_bounds()`
- **Validation**: Automatic bounds checking and enforcement
- **Implementation**: In `base_optimizer.py`, line 47: `self.parameter_bounds = create_10p_parameter_bounds(system_defaults)`

## Common Issues & Solutions

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'calculators'`
**Solution**: Always use `PYTHONPATH=src` when running scripts
```bash
PYTHONPATH=src python src/tests/lattice_results.py
```

### Parameter Bounds Errors
**Problem**: `Default value X for parameter Y not within bounds`
**Solution**: 
1. Check `PARAMETER_CONSTRAINTS` in `parameter_bounds.py`
2. Use `create_10p_parameter_bounds()` for reasonable bounds
3. Validate bounds with `ParameterBoundsManager`

### Parameter Bounds Issues
**Problem**: Poor optimization results due to restrictive bounds
**Diagnosis**:
1. Check if optimal values are within current bounds:
   ```python
   # Check bounds for lattice parameters
   from utils.parameter_bounds import create_10p_parameter_bounds
   from utils.data_extraction import extract_system_parameters
   
   system_defaults = extract_system_parameters(["Cd", "S"])
   bounds = create_10p_parameter_bounds(system_defaults)
   
   # Look for lattice-related parameters
   for bound in bounds:
       if 'lattice' in bound.name.lower() or bound.name in ['a', 'b', 'c']:
           print(f"{bound.name}: [{bound.min_val}, {bound.max_val}]")
   ```
2. If bounds are too restrictive, consider:
   - Using static bounds instead of 10% bounds
   - Increasing the margin factor in `create_10p_parameter_bounds()`
   - Manually adjusting `PARAMETER_CONSTRAINTS`

### Convergence Issues
**Problem**: Poor optimization results
**Solutions**:
1. Check parameter bounds include optimal values
2. Verify reference data is correct
3. Check fitness function weights
4. Ensure proper units (atomic units vs. Ångström)

### Energy Value Issues
**Problem**: Unrealistic energy values (e.g., -0.3 eV instead of -10 eV)
**Solutions**:
1. Check calculator configuration
2. Verify parameter file generation
3. Check units and conversions
4. Validate system configuration

## File-Specific Notes

### `gad.py` (PyGAD Optimizer)
- Uses PyGAD library for genetic algorithm
- Supports parallel processing (default: 8 cores)
- Converts between PyGAD fitness (maximization) and RMSE (minimization)

### `lattice_results.py`
- Analyzes lattice constant optimization results
- Requires `PYTHONPATH=src` for imports
- Generates CSV with optimization results

### `parameter_stats.py`
- Generates histograms and statistics for parameter distributions
- Saves plots to `results/plots/`
- Handles both scalar and array parameters

## Development Tips

### Adding Debug Output
Use `logger.info()` for debug information:
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Parameter bounds: {bounds}")
```

### Testing Individual Components
```bash
# Test parameter bounds
PYTHONPATH=src python -c "from utils.parameter_bounds import ParameterBoundsManager; print('OK')"

# Test calculator
PYTHONPATH=src python -c "from calculators.calc import GeneralCalculator; print('OK')"
```

### Performance Optimization
- Use PyGAD for parallel genetic algorithm optimization
- Consider population size vs. generation count trade-offs
- Monitor failed evaluations count

## Future Improvements

1. **Better Error Handling**: More robust error recovery in optimizers
2. **Parameter Validation**: Enhanced validation of parameter sets
3. **Visualization**: Better plotting and analysis tools
4. **Documentation**: More detailed API documentation
5. **Testing**: Comprehensive unit tests for all components

---

**Last Updated**: [Current Date]
**Maintainer**: [Your Name]
**Project**: TBLite Parameter Optimization 
