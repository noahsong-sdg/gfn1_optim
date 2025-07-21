# TBLite Parameter Optimization Package

This package provides a comprehensive framework for optimizing TBLite parameters using various optimization algorithms.

## Package Structure

```
src/
├── __init__.py              # Main package initialization
├── base_optimizer.py        # Base optimizer class
├── config.py               # System configuration
├── cli.py                  # Unified command-line interface
├── common.py               # Common utilities and configuration
├── optimizers/             # Optimization algorithms
│   ├── __init__.py
│   ├── ga.py              # Genetic Algorithm
│   ├── pso.py             # Particle Swarm Optimization
│   ├── bayes_h.py         # Bayesian Optimization
│   ├── cma1.py            # CMA-ES (cmaes library)
│   └── cma2.py            # CMA-ES (pycma library)
├── calculators/            # Calculation engines
│   ├── __init__.py
│   ├── calc.py            # General calculator
│   └── tblite_ase_calculator.py  # ASE integration
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── data_extraction.py # Parameter extraction
│   └── parameter_bounds.py # Bounds management
└── tests/                  # Testing and validation
    ├── __init__.py
    ├── compare.py         # Method comparison
    ├── bandgap.py         # Bandgap calculations
    └── bulk_materials_validator.py  # Material validation
```

## Key Features

### Centralized Configuration
- **common.py**: Centralized path constants, logging setup, and common imports
- Eliminates redundant code across modules
- Consistent logging configuration

### Unified CLI Interface
- **cli.py**: Single command-line interface for all optimizers
- Consistent parameter handling
- Standardized output format

### Organized Package Structure
- **optimizers/**: All optimization algorithms in one place
- **calculators/**: Calculation engines and ASE integration
- **utils/**: Parameter extraction and bounds management
- **tests/**: Comparison and validation tools

## Usage

### Command Line Interface
```bash
# Run GA optimization
python src/cli.py ga H2 config/gfn1-base.toml --output results/ga

# Run PSO optimization with custom parameters
python src/cli.py pso Si2 config/gfn1-base.toml --generations 100 --output results/pso

# Run Bayesian optimization
python src/cli.py bayes CdS config/gfn1-base.toml --n-calls 200 --output results/bayes
```

### Programmatic Usage
```python
from src import GeneralParameterGA, GAConfig

# Create optimizer
optimizer = GeneralParameterGA(
    system_name="H2",
    base_param_file="config/gfn1-base.toml",
    config=GAConfig(generations=100)
)

# Run optimization
best_params = optimizer.optimize()
```

## Benefits of Refactoring

1. **Eliminated Redundancies**:
   - Removed duplicate `apply_bounds` methods
   - Centralized path constants
   - Unified logging configuration
   - Single CLI interface

2. **Improved Organization**:
   - Logical package structure
   - Clear separation of concerns
   - Better code discoverability

3. **Enhanced Maintainability**:
   - Consistent import patterns
   - Centralized configuration
   - Reduced code duplication

4. **Better Testing**:
   - Organized test modules
   - Clear validation tools
   - Consistent comparison framework

## Migration Notes

- All existing functionality is preserved
- Import paths have been updated to use new package structure
- CLI interface provides backward compatibility
- Checkpoint files remain compatible 
