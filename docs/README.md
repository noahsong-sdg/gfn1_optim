# TBLite Parameter Optimization

A comprehensive framework for optimizing TBLite (Tight-Binding based DFT) parameters using multiple metaheuristic algorithms. The framework supports various molecular systems and provides four different optimization algorithms with a clean, extensible architecture.

## Architecture Overview

The optimization framework uses a **base class inheritance pattern** to eliminate code duplication:

- **BaseOptimizer** (`src/base_optimizer.py`): Contains all common functionality (system config, data handling, fitness evaluation, results saving)
- **Algorithm-specific classes**: Inherit from BaseOptimizer and implement only their core optimization logic
  - `GeneralParameterGA` (Genetic Algorithm)
  - `GeneralParameterPSO` (Particle Swarm Optimization) 
  - `GeneralParameterBayesian` (Bayesian Optimization)
  - `GeneralParameterCMA` (CMA-ES)

## Quick Start

### Prerequisites

```bash
# Core dependencies
pip install numpy pandas toml

# Algorithm-specific dependencies
pip install scikit-optimize  # For Bayesian optimization
pip install cmaes           # For CMA-ES
```

### Basic Usage

```bash
# Run genetic algorithm on H2 system
python src/ga.py H2

# Run particle swarm optimization on Si2 system  
python src/pso.py Si2

# Run Bayesian optimization on any configured system
python src/bayes_h.py C2

# Run CMA-ES optimization
python src/cma.py H2
```

## Running Optimization Algorithms

### 1. Genetic Algorithm (GA)
```bash
python src/ga.py [SYSTEM_NAME]
```

**Configuration options** (edit in `ga.py`):
```python
config = GAConfig(
    population_size=50,     # Number of individuals in population (default: 50)
    generations=100,        # Maximum generations (default: 100)
    mutation_rate=0.1,      # Probability of mutation (default: 0.1)
    crossover_rate=0.8,     # Probability of crossover (default: 0.8)
    tournament_size=3,      # Tournament selection size (default: 3)
    elitism_rate=0.1,       # Fraction of elite individuals preserved (default: 0.1)
    max_workers=4,          # Workers for parallel processing (currently unused)
    convergence_threshold=1e-6,
    patience=20
)
```

### 2. Particle Swarm Optimization (PSO)
```bash
python src/pso.py [SYSTEM_NAME]
```

**Configuration options** (edit in `pso.py`):
```python
config = PSOConfig(
    swarm_size=30,          # Number of particles (default: 30)
    max_iterations=100,     # Maximum iterations (default: 100)
    w=0.7,                  # Inertia weight (default: 0.7)
    c1=1.5,                 # Cognitive coefficient (default: 1.5)
    c2=1.5,                 # Social coefficient (default: 1.5)
    w_min=0.1, w_max=0.9,   # Adaptive inertia range
    use_adaptive_inertia=True,  # Use adaptive inertia (default: True)
    convergence_threshold=1e-6,
    patience=20
)
```

### 3. Bayesian Optimization
```bash
python src/bayes_h.py [SYSTEM_NAME]
```

**Configuration options** (edit in `bayes_h.py`):
```python
config = BayesianConfig(
    n_calls=100,            # Number of function evaluations (default: 100)
    n_initial_points=10,    # Random points to start with (default: 10)
    acq_func="EI",          # Acquisition function: "EI", "LCB", "PI" (default: "EI")
    xi=0.01,                # Exploration-exploitation trade-off (default: 0.01)
    kappa=1.96,             # Lower confidence bound parameter (default: 1.96)
    n_restarts_optimizer=5, # Acquisition optimization restarts (default: 5)
    random_state=None       # Random state for reproducibility (default: None)
)
```

### 4. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
```bash
python src/cma.py [SYSTEM_NAME]
```

**Configuration options** (edit in `cma.py`):
```python
config = CMAConfig(
    sigma=0.5,              # Initial step size (default: 0.5)
    max_generations=100,    # Maximum generations (default: 100)
    population_size=None,   # Use CMA-ES default (4 + 3*log(dim))
    bounds_handling="repair", # "repair" or "penalty" (default: "repair")
    seed=None,              # Random seed for reproducibility (default: None)
    convergence_threshold=1e-6,
    patience=20
)
```

## Adding a New System

### Step 1: Add System Configuration

Edit `src/config.py` and add your system to the `get_system_config()` function:

```python
def get_system_config(system_name: str) -> SystemConfig:
    """Get system-specific configuration"""
    
    if system_name == "YourSystem":
        return SystemConfig(
            name="YourSystem",
            elements=["Element1", "Element2"],  # e.g., ["C", "N"] for CN
            geometry_file="geometries/yoursystem.xyz",
            reference_data_file="results/curves/yoursystem_reference.csv",
            spin=0,  # 0 for singlet, 1 for doublet, 2 for triplet, etc.
            charge=0,
            # Output file locations
            optimized_params_file="results/parameters/yoursystem_optimized.toml",
            fitness_history_file="results/fitness/yoursystem_fitness_history.csv"
        )
```

### Step 2: Create Reference Data

You need high-quality reference data (typically from CCSD(T) or experimental sources):

```bash
# Create the reference file
mkdir -p results/curves
```

Create `results/curves/yoursystem_reference.csv` with columns:
```csv
Distance,Energy
1.0,-100.523
1.1,-100.834
1.2,-101.045
...
```

### Step 3: Create System Geometry

Create `geometries/yoursystem.xyz`:
```
2
YourSystem molecule
Element1  0.0  0.0  0.0
Element2  1.0  0.0  0.0
```

### Step 4: Test the New System

```bash
python src/ga.py YourSystem
```

## Adding a New Fitness Function

The fitness function is implemented in `BaseOptimizer.evaluate_fitness()`. To add a new fitness function:

### Option 1: Modify Base Class (affects all algorithms)

Edit `src/base_optimizer.py`, find the `evaluate_fitness()` method:

```python
def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
    """Evaluate fitness of parameters by calculating curve error"""
    try:
        # ... existing code ...
        
        # Replace this section with your new fitness function:
        # Current: RMSE between reference and calculated energies
        rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
        
        # Example alternatives:
        # 1. Mean Absolute Error
        # mae = np.mean(np.abs(ref_relative - calc_relative))
        # return mae
        
        # 2. Weighted RMSE (emphasize certain distance ranges)
        # weights = np.exp(-((distances - 2.0) ** 2) / 0.5)  # Weight around 2.0 Å
        # weighted_rmse = np.sqrt(np.mean(weights * (ref_relative - calc_relative)**2))
        # return weighted_rmse
        
        # 3. Multi-objective: RMSE + parameter regularization
        # param_penalty = sum(abs(p - default) for p, default in zip(parameters.values(), defaults))
        # return rmse + 0.1 * param_penalty
        
        return rmse
        
    except Exception as e:
        # ... error handling ...
```

### Option 2: Create Algorithm-Specific Fitness (advanced)

Override the fitness evaluation in specific algorithm classes:

```python
# In src/ga.py, add to GeneralParameterGA class:
def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
    """Custom fitness function for GA"""
    # Call base fitness
    base_fitness = super().evaluate_fitness(parameters)
    
    # Add GA-specific modifications
    # Example: Penalize complex parameter sets
    complexity_penalty = len([p for p in parameters.values() if abs(p) > 10.0])
    
    return base_fitness + 0.01 * complexity_penalty
```

### Available Fitness Function Components

The base fitness evaluation provides access to:

- `ref_energies`: Reference energy values
- `calc_energies`: Calculated energy values  
- `distances`: Molecular distances
- `parameters`: Current parameter set
- `self.system_config`: System configuration

Common fitness functions you can implement:

1. **RMSE** (current default): `sqrt(mean((ref - calc)^2))`
2. **MAE**: `mean(|ref - calc|)`
3. **Max Error**: `max(|ref - calc|)`
4. **Weighted RMSE**: Emphasize specific distance ranges
5. **Multi-objective**: Combine accuracy + parameter complexity
6. **Physical constraints**: Penalize unphysical parameters

## Output Files

Each optimization run generates:

- **Optimized parameters**: `results/parameters/{system}_optimized.toml`
- **Fitness history**: `results/fitness/{system}_fitness_history.csv`
- **Algorithm-specific results**: Various formats depending on the algorithm

## Advanced Usage

### Custom Parameter Bounds

Modify parameter bounds in `src/data_extraction.py` or override in the base optimizer:

```python
# Example: Tighter bounds for specific parameters
def _get_parameter_bounds(self, param_name: str, default_val: float):
    if param_name == "element.C.levels[0]":
        return (-15.0, -10.0)  # Custom range for carbon 2s level
    # ... use default logic for other parameters
```

### Parallelization & Workers

**Current Worker Configuration:**

| Algorithm | Parallel Processing | Workers | Notes |
|-----------|-------------------|---------|--------|
| **GA** | ❌ Not implemented | `max_workers=4` (unused) | Sequential fitness evaluation |
| **PSO** | ❌ Not implemented | N/A | Sequential particle evaluation |
| **Bayesian** | ❌ Not implemented | N/A | Inherently sequential algorithm |
| **CMA-ES** | ❌ Not implemented | N/A | Sequential individual evaluation |

**⚠️ Important Note:** Although the GA configuration includes `max_workers=4`, the current implementation does **not** use parallel processing. All algorithms evaluate fitness sequentially, which may be slower for systems with many parameters.

**Performance Considerations:**
```python
# For better performance with large parameter spaces:
config = GAConfig(
    population_size=100,    # Larger populations for better exploration
    generations=200,        # More generations for convergence
    max_workers=4          # Currently unused, but planned for future
)

config = PSOConfig(
    swarm_size=50,         # Larger swarms for complex landscapes
    max_iterations=200     # More iterations for convergence
)
```

**Future Enhancement:** Parallel fitness evaluation could significantly speed up optimization, especially for:
- Large parameter spaces (>20 parameters)
- Complex molecular systems
- High-resolution dissociation curves

### Convergence Tuning

Adjust convergence criteria based on your system:

```python
config = GAConfig(
    convergence_threshold=1e-8,  # Stricter convergence
    patience=50                  # Wait longer for improvements
)
```

## Troubleshooting

### Common Issues

1. **SCF Convergence Failures**: 
   - Reduce parameter bounds
   - Check Slater exponent values (must be > 0.5)
   - Verify geometry file format

2. **Poor Optimization Results**:
   - Increase population size/iterations
   - Check reference data quality
   - Verify parameter bounds are reasonable

3. **Import Errors**:
   ```bash
   pip install scikit-optimize cmaes
   ```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new features:

1. **New algorithms**: Inherit from `BaseOptimizer`
2. **New systems**: Add to `config.py`
3. **New fitness functions**: Modify `evaluate_fitness()` method
4. **Keep it modular**: Don't duplicate code between algorithm implementations

## Citation

If you use this optimization framework in your research, please cite the relevant algorithm papers and the TBLite method. 
