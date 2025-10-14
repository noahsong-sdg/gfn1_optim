# TBLite Parameter Optimization 

A module designed for experimentation on GFN1-xTB, a generalized tight binding method. The workflow entails selecting a task, creating a dataset, then optimizing the backend parameters of GFN1-xTB with various meta-heuristic algorithms.

This work is done with support from the Eaves group! It is very much still in progress. 

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
│   ├── cma1.py            # CMA-ES (cmaes)
│   └── cma2.py            # CMA-ES (pycma)
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

## Usage

### Command Line Interface
```bash
python src/cli.py ga H2 config/gfn1-base.toml --output results/ga

python src/cli.py pso Si2 config/gfn1-base.toml --generations 100 --output results/pso

python src/cli.py bayes CdS config/gfn1-base.toml --n-calls 200 --output results/bayes
```

### Programmatic Use
```python
from src import GeneralParameterGA, GAConfig

optimizer = GeneralParameterGA(
    system_name="H2",
    base_param_file="config/gfn1-base.toml",
    config=GAConfig(generations=100)
)

best_params = optimizer.optimize()
```
