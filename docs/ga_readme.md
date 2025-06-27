# things to add
- fn to extract params rather than manually typing them
- 



# Genetic Algorithm for TBLite Parameter Optimization

A genetic algorithm system for optimizing GFN1-xTB parameters in TBLite, specifically designed for improving H₂ dissociation curve accuracy compared to CCSD reference data.

## Overview

This system uses evolutionary optimization to find better parameter combinations for the GFN1-xTB Hamiltonian than the default values or optimization (NEWUOA). It focuses on 16 H₂-relevant parameters and optimizes them to minimize the RMSE between calculated and CCSD reference H₂ dissociation curves.

## Files Structure

```
src/
├── ga.py              # Core genetic algorithm implementation
├── ga_analysis.py     # Results analysis and visualization
├── test_ga.py         # Interactive test suite and examples
└── h2_v2.py          # Existing H₂ calculation infrastructure (unchanged)

# Generated during optimization:
├── ga_optimized_params.toml    # Best parameters found by GA
├── ga_fitness_history.csv      # Evolution progress over generations
├── h2_ccsd_data.csv           # CCSD reference data (auto-generated)
└── ga_analysis_results/       # Analysis plots and reports
```

## Prerequisites

1. **TBLite environment**: Ensure `pixi run tblite` works
2. **Base parameters**: `gfn1-base.toml` must be present
3. **Python dependencies**: Install required packages:
   ```bash
   pixi add toml numpy pandas matplotlib seaborn
   ```

## Quick Start

### 1. Test the System
```bash
cd src
python test_ga.py
# if you have pixi:
pixi run python test_ga.py
# Select option 1: Quick test (8 individuals, 5 generations)
```

This runs a minimal test to verify everything works.

### 2. Full Optimization
```bash
cd src  
python test_ga.py
# Select option 2: Full optimization (30 individuals, 50 generations)
```

This runs the real optimization (may take 2-6 hours).

### 3. Analyze Results
```bash
cd src
python test_ga.py  
# Select option 3: Analyze existing results
```

Generates comprehensive analysis plots and statistics.

## Detailed Workflow

### Phase 1: Initialization
1. **Load base parameters** from `gfn1-base.toml`
2. **Define optimization targets**: 16 H₂-relevant parameters:
   - Hamiltonian parameters: `kpol`, `enscale`, shell interactions
   - H-H pair interaction: `hamiltonian.xtb.kpair.H-H`
   - Hydrogen element properties: orbital energies, Slater exponents, etc.
3. **Generate reference data**: Calculate CCSD/cc-pVTZ H₂ curve (50 points, 0.5-2.5 Å)
4. **Initialize population**: Create 30-50 random parameter sets within physical bounds

### Phase 2: Evolution Loop
For each generation (up to 50):

1. **Parallel fitness evaluation**:
   ```python
   for individual in population:
       # Create temporary parameter file with individual's values
       # Run TBLite H₂ calculations at 50 distances  
       # Compare with CCSD reference using RMSE
       # fitness = 1/(1 + RMSE)
   ```

2. **Selection**: Tournament selection (pick best of 3 random individuals)

3. **Crossover**: Uniform parameter mixing between parents (80% rate)

4. **Mutation**: Gaussian noise addition to parameters (15% rate)

5. **Elitism**: Keep top 10% of population

6. **Convergence check**: Stop if no improvement for 15 generations

### Phase 3: Analysis
1. **Fitness evolution**: Plot improvement over generations
2. **H₂ curve comparison**: Compare original vs GA-optimized vs CCSD
3. **Parameter analysis**: Identify which parameters changed most
4. **Performance metrics**: Calculate RMSE improvement percentage

## Expected Results

### Fitness Evolution
- Should show steady improvement over first 10-20 generations
- Convergence typically occurs around generation 30-40
- Final fitness should be significantly higher than initial

### H₂ Curve Accuracy
- **Original GFN1-xTB**: Typical RMSE ~0.01-0.05 Hartree vs CCSD
- **GA-optimized**: Expected 20-50% RMSE reduction
- **Curve shape**: Better reproduction of CCSD minimum and dissociation

### Parameter Changes
Most commonly optimized parameters:
- `element.H.levels[0]` (1s orbital energy): ±10-20% change
- `hamiltonian.xtb.kpair.H-H` (pair interaction): ±5-15% change  
- `element.H.gam` (gamma parameter): ±10-30% change
- `element.H.zeff` (effective nuclear charge): ±5-15% change

## Configuration Options

### GA Parameters (in `GAConfig`)
```python
population_size: int = 30     # More individuals = better exploration
generations: int = 50         # More generations = longer optimization  
mutation_rate: float = 0.15   # Higher = more exploration vs exploitation
crossover_rate: float = 0.8   # Higher = more parameter mixing
max_workers: int = 4          # Parallel evaluations (match CPU cores)
```

### Parameter Bounds
Each parameter has physically reasonable bounds:
```python
"hamiltonian.xtb.kpol": (1.0, 5.0)           # Coulomb parameter
"element.H.levels[0]": (-15.0, -8.0)         # 1s orbital energy  
"element.H.zeff": (0.8, 1.5)                 # Effective nuclear charge
```

## Output Files

### `ga_optimized_params.toml`
Complete parameter file with optimized values, ready to use with TBLite:
```bash
pixi run tblite run --method gfn1 --param ga_optimized_params.toml coord
```

### `ga_fitness_history.csv`
Evolution tracking with columns:
- `generation`: Generation number
- `best_fitness`: Best individual fitness this generation
- `avg_fitness`: Population average fitness
- `std_fitness`: Population fitness diversity

### `ga_analysis_results/`
- `fitness_evolution.png`: Fitness improvement over time
- `h2_curve_comparison.png`: H₂ curves (CCSD vs original vs GA-optimized)
- `parameter_changes.png`: Which parameters changed most
- `convergence_analysis.png`: Convergence characteristics
- `parameter_changes.csv`: Detailed parameter change statistics

## Comparison with NEWUOA (tblite fit)

| Aspect | NEWUOA (tblite fit) | Genetic Algorithm |
|--------|---------------------|-------------------|
| **Method** | Derivative-free local optimization | Population-based global optimization |
| **Evaluations** | ~100-500 | ~1500-3000 |
| **Time** | 10+ hours (single core) | 2-6 hours (4 cores) |
| **Convergence** | To local minimum | To global optimum (potentially) |
| **Robustness** | Can get stuck | Explores parameter space broadly |
| **Parameters** | Can optimize all ~2500 | Focuses on 16 H₂-relevant parameters |

## Troubleshooting

### Common Issues

1. **"No module named 'toml'"**:
   ```bash
   pixi add toml
   ```

2. **TBLite command fails**:
   - Ensure `pixi run tblite --help` works
   - Check that `gfn1-base.toml` exists

3. **Out of memory errors**:
   - Reduce `population_size` and `max_workers`
   - Use fewer H₂ test points (modify `test_distances`)

4. **Convergence issues**:
   - Increase `mutation_rate` for more exploration
   - Increase `patience` parameter
   - Check parameter bounds are reasonable

### Performance Optimization

1. **Faster evaluation**:
   - Increase `max_workers` (up to CPU cores)
   - Reduce number of H₂ test points
   - Use smaller population for testing

2. **Better convergence**:
   - Increase population size
   - Tune mutation/crossover rates
   - Start with parameters from previous optimization

## Advanced Usage

### Custom Parameter Sets
Modify `_define_h2_parameter_bounds()` to optimize different parameters:
```python
bounds.append(ParameterBounds("element.C.gam", 0.3, 0.7, 0.48))  # Add carbon
```

### Different Molecules
Extend beyond H₂ by modifying:
- Test geometries in fitness function
- Parameter bounds for relevant elements
- Reference data source

### Hybrid Optimization
```python
# Phase 1: GA for global search
best_individual = ga.optimize()

# Phase 2: Local refinement with NEWUOA
ga.save_best_parameters("ga_start.toml")
subprocess.run(["pixi", "run", "tblite", "fit", "ga_start.toml", "tbfit.toml"])
```

## Scientific Context

### Why Genetic Algorithms for Parameter Optimization?

1. **Global optimization**: Can escape local minima that gradient methods miss
2. **Parameter coupling**: Naturally handles interdependent parameters
3. **Robustness**: Less sensitive to initial parameter values
4. **Interpretability**: Easy to analyze which parameters matter most

### H₂ as a Test Case

H₂ is ideal for parameter optimization because:
- Simple system with accurate CCSD reference available
- Sensitive to parameter quality
- Fast to calculate (enables many evaluations)
- Representative of chemical bonding physics

### Applications Beyond H₂

The optimized parameters should improve:
- Other hydrogen-containing molecules
- Bond dissociation energies
- Reaction barrier heights
- Thermochemistry accuracy

## References

- GFN1-xTB method: Grimme et al., J. Chem. Theory Comput. 2017, 13, 1989-2009
- TBLite implementation: https://tblite.readthedocs.io/
- Genetic algorithms: Holland, J.H. "Adaptation in Natural and Artificial Systems" (1992) 
