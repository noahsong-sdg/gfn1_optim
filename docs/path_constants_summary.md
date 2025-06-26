# Path Constants System

## Overview
All optimization files now use portable path constants defined at the top of each file. This makes it easy to adapt the code to different molecular systems or clusters.

## Implementation Pattern

Each file now follows this pattern:

```python
from pathlib import Path

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

# Reference data files
CCSD_REFERENCE_DATA = RESULTS_DIR / "curves" / "h2_ccsd_data.csv"

# Output files
OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "method_optimized_params.toml"
FITNESS_HISTORY = RESULTS_DIR / "fitness" / "method_fitness_history.csv"
```

## Key Benefits

1. **Cluster Portable**: Uses `Path.cwd()` instead of hardcoded absolute paths
2. **Easy Adaptation**: Change constants at top of file to adapt to new molecular systems
3. **Consistent Structure**: All methods follow same directory conventions
4. **Clear Separation**: Config, data, and results are cleanly separated

## Files Updated

- `src/h2_v2.py` - Main comparison framework
- `src/bayesian.py` - Bayesian optimization
- `src/bayesian_test.py` - Bayesian test suite
- `src/ga.py` - Genetic algorithm
- `src/pso.py` - Particle swarm optimization

## Adapting to New Systems

To adapt for a different molecular system (e.g., CO2):

1. Update the reference data constants:
   ```python
   CCSD_REFERENCE_DATA = RESULTS_DIR / "curves" / "co2_ccsd_data.csv"
   ```

2. Update output file names:
   ```python
   BAYESIAN_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "co2_bayesian_optimized_params.toml"
   ```

3. Update parameter bounds in the `_define_parameter_bounds()` methods to include relevant elements (C, O)

4. Update fitness evaluation to use appropriate reference calculations

## Directory Structure

```
project_root/
├── config/           # Base parameter files
├── results/
│   ├── curves/      # Dissociation curve data
│   ├── parameters/  # Optimized parameter files
│   ├── fitness/     # Fitness history files
│   └── plots/       # Visualization outputs
├── data/            # Input/reference data
└── src/             # Source code
``` 
