import subprocess
import numpy as np
import random
import copy
import toml
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
from scipy.interpolate import interp1d

from calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from data_extraction import GFN1ParameterExtractor
from config import get_system_config, SystemConfig, get_calculation_distances

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

@dataclass
class GAConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_rate: float = 0.1
    mutation_strength: float = 0.05
    max_workers: int = 4
    convergence_threshold: float = 1e-6
    patience: int = 20

@dataclass
class ParameterBounds:
    """Bounds for parameter optimization"""
    name: str
    min_val: float
    max_val: float
    default_val: float

class Individual:
    """Represents a single parameter set (genome) in the GA population"""
    
    def __init__(self, parameters: Dict[str, float], fitness: float = 0.0):
        self.parameters = parameters.copy()
        self.fitness = fitness
        self.age = 0
        
    def copy(self):
        """Create a deep copy of the individual"""
        return Individual(self.parameters.copy(), self.fitness)
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.6f}, age={self.age})"

class GeneralParameterGA:
    """General Genetic Algorithm for optimizing TBLite parameters for any molecular system"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: GAConfig = GAConfig(),
                 train_fraction: float = 0.8):
        """Initialize the genetic algorithm for TBLite parameter optimization
        
        Args:
            system_name: Name of the system to optimize (e.g., 'H2', 'Si2', 'CdS')
            base_param_file: Path to base parameter TOML file
            reference_data: Optional reference data (if None, generates with method)
            config: GA configuration
            train_fraction: Fraction of data to use for training (rest for testing)
        """
        
        # System configuration
        self.system_name = system_name
        self.system_config = get_system_config(system_name)
        
        # Load base parameters
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        
        # Store base parameter file path for parameter extraction
        self.base_param_file = Path(base_param_file)
        
        # Configuration
        self.config = config
        self.train_fraction = train_fraction
        
        # Define parameter bounds using parameter extraction
        self.parameter_bounds = self._define_parameter_bounds()
        
        # Load or generate reference data
        if reference_data is not None:
            self.full_reference_data = reference_data
        else:
            self.full_reference_data = self._load_or_generate_reference_data()
        
        # Split data into train/test sets
        self._split_train_test_data()
        
        # Initialize optimization state
        self.population = []
        self.best_individual = None
        self.generation = 0
        self.convergence_counter = 0
        self.fitness_history = []
        self.failed_evaluations = 0
        
    def _define_parameter_bounds(self) -> List[ParameterBounds]:
        """Define parameter bounds for system-relevant parameters using extracted defaults"""
        bounds = []
        
        # Extract default parameters from base parameter file
        extractor = GFN1ParameterExtractor(self.base_param_file)
        system_defaults = extractor.extract_defaults_dict(self.system_config.elements)
        
        # Define bounds based on parameter type and extracted defaults
        for param_name, default_val in system_defaults.items():
            min_val, max_val = self._get_parameter_bounds(param_name, default_val)
            bounds.append(ParameterBounds(param_name, min_val, max_val, default_val))
        
        logger.info(f"Generated {len(bounds)} parameter bounds for {self.system_name} from extracted defaults")
        return bounds
    
    def _get_parameter_bounds(self, param_name: str, default_val: float) -> tuple:
        """Get appropriate bounds for a parameter based on its name and default value"""
        
        # System-specific parameter bounds (can be extended for different systems)
        system_specific_bounds = {
            'H2': {
                'hamiltonian.xtb.kpair.H-H': (0.5, 1.5),
            },
            'Si2': {
                'hamiltonian.xtb.kpair.Si-Si': (0.5, 1.5),
            },
            'C2': {
                'hamiltonian.xtb.kpair.C-C': (0.5, 1.5),
            },
            'N2': {
                'hamiltonian.xtb.kpair.N-N': (0.5, 1.5),
            },
            'O2': {
                'hamiltonian.xtb.kpair.O-O': (0.5, 1.5),
            },
        }
        
        # Check for system-specific bounds first
        if self.system_name in system_specific_bounds:
            if param_name in system_specific_bounds[self.system_name]:
                return system_specific_bounds[self.system_name][param_name]
        
        # General parameter bounds
        general_bounds = {
            'hamiltonian.xtb.kpol': (1.0, 5.0),
            'hamiltonian.xtb.enscale': (-0.02, 0.02),
            'hamiltonian.xtb.shell.ss': (1.0, 3.0),
            'hamiltonian.xtb.shell.pp': (1.5, 3.5),
            'hamiltonian.xtb.shell.sp': (1.5, 3.0),
        }
        
        if param_name in general_bounds:
            return general_bounds[param_name]
        
        # Type-based bounds
        if 'levels' in param_name:
            # Energy levels - allow ±30% variation
            margin = abs(default_val) * 0.3
            if margin < 1e-6:  # Handle near-zero defaults
                margin = 0.1
            return (default_val - margin, default_val + margin)
        elif 'slater' in param_name:
            # Slater exponents - keep positive, allow wide range
            min_val = max(0.1, default_val * 0.5)
            max_val = default_val * 2.0
            return (min_val, max(max_val, min_val + 0.1))  # Ensure max > min
        elif 'kcn' in param_name:
            # Coordination number parameters - keep positive, allow wide range
            min_val = max(0.001, default_val * 0.1)
            max_val = default_val * 5.0
            return (min_val, max(max_val, min_val + 0.1))  # Ensure max > min
        elif param_name.endswith('.gam'):
            return (0.2, 0.8)
        elif param_name.endswith('.zeff'):
            return (0.8, 1.5)
        elif param_name.endswith('.arep'):
            return (1.5, 3.0)
        elif param_name.endswith('.en'):
            return (1.5, 3.0)
        else:
            # Default: ±50% around default
            margin = abs(default_val) * 0.5
            if margin < 1e-6:  # Handle near-zero defaults
                margin = 0.1
            min_val = default_val - margin
            max_val = default_val + margin
            
            # Final validation to ensure valid bounds
            if max_val <= min_val:
                logger.warning(f"Computed invalid bounds for {param_name} (default={default_val}). Using fallback.")
                return (default_val - 0.1, default_val + 0.1)
            
            return (min_val, max_val)
    
    def _load_or_generate_reference_data(self) -> pd.DataFrame:
        """Load or generate reference data for the system"""
        # For H2, always use CCSD reference data
        if self.system_name == "H2":
            ccsd_file = RESULTS_DIR / "curves" / "h2_ccsd_500.csv"
            if ccsd_file.exists():
                logger.info(f"Loading CCSD reference data from {ccsd_file}")
                return pd.read_csv(ccsd_file)
            else:
                raise FileNotFoundError(f"CCSD reference file {ccsd_file} not found. "
                                      "GA requires CCSD data for meaningful optimization.")
        
        # For other systems, try system-specific reference file
        ref_file = Path(self.system_config.reference_data_file)
        if ref_file.exists():
            logger.info(f"Loading reference data from {ref_file}")
            return pd.read_csv(ref_file)
        else:
            logger.warning(f"No reference data found for {self.system_name}. "
                          f"Generating with GFN1-xTB (not recommended for optimization).")
            # Generate with GFN1-xTB as fallback
            calc_config = CalcConfig(method=CalcMethod.GFN1_XTB)
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            ref_data = generator.generate_curve(
                save=True, filename=str(ref_file)
            )
            return ref_data
    
    def _set_parameter_in_dict(self, param_dict: dict, path: str, value: float):
        """Set a parameter value using path like 'element.H.levels[0]' or 'hamiltonian.xtb.kpol'"""
        import re
        
        # Check if this is an array access
        if '[' in path and ']' in path:
            # Split into path and array index: 'element.H.levels[0]' -> 'element.H.levels', '0'
            match = re.match(r'(.+)\[(\d+)\]$', path)
            if match:
                array_path, index_str = match.groups()
                index = int(index_str)
                
                # Navigate to the array location
                keys = array_path.split('.')
                current = param_dict
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the array element
                array_name = keys[-1]
                if array_name not in current:
                    current[array_name] = []
                while len(current[array_name]) <= index:
                    current[array_name].append(0.0)
                current[array_name][index] = value
                return
        
        # Regular path access
        keys = path.split('.')
        current = param_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def create_individual(self, parameters: Optional[Dict[str, float]] = None) -> Individual:
        """Create a new individual with given or random parameters"""
        if parameters is None:
            # Generate random parameters within bounds
            parameters = {}
            for bound in self.parameter_bounds:
                # Validate bounds
                if bound.max_val <= bound.min_val:
                    logger.warning(f"Invalid bounds for {bound.name}: min={bound.min_val}, max={bound.max_val}. Using default.")
                    parameters[bound.name] = bound.default_val
                    continue
                
                if random.random() < 0.8:  # 80% chance to stay near default
                    # Normal distribution around default
                    range_size = bound.max_val - bound.min_val
                    std = max(range_size * 0.1, 1e-6)  # Ensure positive std with minimum value
                    value = np.random.normal(bound.default_val, std)
                else:
                    # Uniform distribution across full range
                    value = random.uniform(bound.min_val, bound.max_val)
                
                # Clamp to bounds
                value = max(bound.min_val, min(bound.max_val, value))
                parameters[bound.name] = value
        
        return Individual(parameters)
    
    def apply_bounds(self, individual: Individual):
        """Apply parameter bounds to an individual"""
        for bound in self.parameter_bounds:
            if bound.name in individual.parameters:
                value = individual.parameters[bound.name]
                individual.parameters[bound.name] = max(
                    bound.min_val, min(bound.max_val, value)
                )
    
    def create_param_file(self, individual: Individual) -> str:
        """Create a temporary parameter file for an individual"""
        # Start with base parameters
        params = copy.deepcopy(self.base_params)
        
        # Apply individual's parameters
        for param_name, value in individual.parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(params, f)
            return f.name
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate fitness of an individual by calculating curve error on training data"""
        try:
            param_file = self.create_param_file(individual)
            
            # Create calculator with custom parameters
            calc_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file
            )
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Calculate curve on TRAINING distances only
            calc_data = generator.generate_curve(self.train_distances)
            
            # Clean up temp file
            os.unlink(param_file)
            
            # Compare with TRAINING reference data
            ref_energies = self.reference_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            # Verify shapes match
            if len(ref_energies) != len(calc_energies):
                raise ValueError(f"Shape mismatch: reference {len(ref_energies)} vs calculated {len(calc_energies)}")
            
            # Convert to relative energies
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            # Calculate RMSE on training data
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            
            # Fitness is inverse of error (higher is better)
            fitness = 1.0 / (1.0 + rmse)
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            self.failed_evaluations += 1
            return 0.0
    
    def evaluate_test_performance(self, individual: Individual) -> Dict[str, float]:
        """Evaluate an individual's performance on the test set"""
        try:
            param_file = self.create_param_file(individual)
            
            # Create calculator with custom parameters
            calc_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file
            )
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Calculate curve on TEST distances
            calc_data = generator.generate_curve(self.test_reference_data['Distance'].values)
            
            # Clean up temp file
            os.unlink(param_file)
            
            # Compare with TEST reference data
            ref_energies = self.test_reference_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            # Convert to relative energies
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            mae = np.mean(np.abs(ref_relative - calc_relative))
            max_error = np.max(np.abs(ref_relative - calc_relative))
            
            return {
                'test_rmse': rmse,
                'test_mae': mae,
                'test_max_error': max_error,
                'test_fitness': 1.0 / (1.0 + rmse)
            }
            
        except Exception as e:
            logger.warning(f"Test evaluation failed: {e}")
            return {
                'test_rmse': float('inf'),
                'test_mae': float('inf'),
                'test_max_error': float('inf'),
                'test_fitness': 0.0
            }
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1_params = {}
        child2_params = {}
        
        for param_name in parent1.parameters:
            if random.random() < 0.5:
                child1_params[param_name] = parent1.parameters[param_name]
                child2_params[param_name] = parent2.parameters[param_name]
            else:
                child1_params[param_name] = parent2.parameters[param_name]
                child2_params[param_name] = parent1.parameters[param_name]
        
        child1 = Individual(child1_params)
        child2 = Individual(child2_params)
        
        return child1, child2
    
    def arithmetic_crossover(self, parent1: Individual, parent2: Individual, alpha: float = 0.5) -> Tuple[Individual, Individual]:
        """Arithmetic crossover - creates intermediate solutions by blending parameters"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1_params = {}
        child2_params = {}
        
        for param_name in parent1.parameters:
            p1_val = parent1.parameters[param_name]
            p2_val = parent2.parameters[param_name]
            
            # Linear combination with random alpha for each parameter
            local_alpha = random.uniform(0.3, 0.7)  # Avoid extreme blends
            
            child1_params[param_name] = local_alpha * p1_val + (1 - local_alpha) * p2_val
            child2_params[param_name] = (1 - local_alpha) * p1_val + local_alpha * p2_val
        
        child1 = Individual(child1_params)
        child2 = Individual(child2_params)
        
        # Apply bounds to ensure valid parameters
        self.apply_bounds(child1)
        self.apply_bounds(child2)
        
        return child1, child2
    
    def mutate(self, individual: Individual):
        """Gaussian mutation"""
        for param_name in individual.parameters:
            if random.random() < self.config.mutation_rate:
                bound = next(b for b in self.parameter_bounds if b.name == param_name)
                
                # Gaussian mutation with adaptive strength
                mutation_std = (bound.max_val - bound.min_val) * self.config.mutation_strength
                mutation = np.random.normal(0, mutation_std)
                
                individual.parameters[param_name] += mutation
        
        self.apply_bounds(individual)
    
    def initialize_population(self):
        """Initialize the population"""
        logger.info(f"Initializing population of size {self.config.population_size}")
        
        # Include one individual with default parameters
        default_params = {bound.name: bound.default_val for bound in self.parameter_bounds}
        self.population = [self.create_individual(default_params)]
        
        # Add random individuals
        for _ in range(self.config.population_size - 1):
            self.population.append(self.create_individual())
    
    def evaluate_population_parallel(self):
        """Evaluate population fitness in parallel"""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all fitness evaluations
            future_to_individual = {
                executor.submit(self.evaluate_fitness, individual): individual
                for individual in self.population
            }
            
            # Collect results
            for future in as_completed(future_to_individual):
                individual = future_to_individual[future]
                try:
                    individual.fitness = future.result()
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    individual.fitness = 0.0
                    self.failed_evaluations += 1
        
        # Check for systematic failures
        zero_fitness_count = sum(1 for ind in self.population if ind.fitness == 0.0)
        if zero_fitness_count >= len(self.population) * 0.8:  # 80% failed
            raise RuntimeError(f"Too many fitness evaluation failures: {zero_fitness_count}/{len(self.population)}")
    
    def evolve_generation(self):
        """Evolve one generation"""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best individual
        current_best = self.population[0]
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
            self.convergence_counter = 0
        else:
            self.convergence_counter += 1
        
        # Record fitness history
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': current_best.fitness,
            'avg_fitness': np.mean([ind.fitness for ind in self.population]),
            'std_fitness': np.std([ind.fitness for ind in self.population])
        })
        
        # Elitism: keep best individuals
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        new_population = self.population[:elite_count].copy()
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
        
        # Age the population
        for individual in self.population:
            individual.age += 1
    
    def check_convergence(self) -> bool:
        """Check if the algorithm has converged"""
        if len(self.fitness_history) < 2:
            return False
        
        # Check if improvement is below threshold
        recent_improvement = (
            self.fitness_history[-1]['best_fitness'] - 
            self.fitness_history[-2]['best_fitness']
        )
        
        if abs(recent_improvement) < self.config.convergence_threshold:
            return self.convergence_counter >= self.config.patience
        
        return False
    
    def optimize(self) -> Individual:
        """Run the genetic algorithm optimization"""
        logger.info(f"Starting genetic algorithm optimization for {self.system_name}")
        logger.info(f"Using {len(self.train_distances)} training points for optimization")
        start_time = time.time()
        
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.config.generations):
            self.generation = generation
            
            logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate fitness
            gen_start = time.time()
            try:
                self.evaluate_population_parallel()
            except RuntimeError as e:
                logger.error(f"Stopping optimization due to systematic failures: {e}")
                break
            
            eval_time = time.time() - gen_start
            
            # Log progress
            best_fitness = max(ind.fitness for ind in self.population)
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            
            logger.info(f"  Best fitness: {best_fitness:.6f}")
            logger.info(f"  Avg fitness: {avg_fitness:.6f}")
            logger.info(f"  Failed evaluations: {self.failed_evaluations}")
            logger.info(f"  Evaluation time: {eval_time:.2f}s")
            
            # Early stopping if all fitness values are 0
            if best_fitness == 0.0:
                logger.error("All individuals have zero fitness - stopping optimization")
                break
            
            # Check convergence
            if self.check_convergence():
                logger.info(f"Converged at generation {generation + 1}")
                break
            
            # Evolve
            self.evolve_generation()
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        logger.info(f"Total failed evaluations: {self.failed_evaluations}")
        
        # Evaluate best individual on test set
        if self.best_individual is not None:
            test_results = self.evaluate_test_performance(self.best_individual)
            logger.info("Test set performance:")
            logger.info(f"  Test RMSE: {test_results['test_rmse']:.6f}")
            logger.info(f"  Test MAE: {test_results['test_mae']:.6f}")
            logger.info(f"  Test Max Error: {test_results['test_max_error']:.6f}")
            logger.info(f"  Test Fitness: {test_results['test_fitness']:.6f}")
        
        return self.best_individual
    
    def save_best_parameters(self, filename: str):
        """Save the best parameters to a TOML file"""
        if self.best_individual is None:
            raise ValueError("No optimization has been run")
        
        # Create full parameter set
        params = copy.deepcopy(self.base_params)
        for param_name, value in self.best_individual.parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        with open(filename, 'w') as f:
            toml.dump(params, f)
        
        logger.info(f"Best parameters saved to {filename}")
    
    def save_fitness_history(self, filename: str):
        """Save fitness history to CSV"""
        if not self.fitness_history:
            raise ValueError("No optimization has been run")
        
        df = pd.DataFrame(self.fitness_history)
        df.to_csv(filename, index=False)
        logger.info(f"Fitness history saved to {filename}")

    def _split_train_test_data(self):
        """Split reference data into training and test sets using random sampling"""
        # Get distance and energy columns from full dataset
        if 'Distance' in self.full_reference_data.columns:
            full_distances = self.full_reference_data['Distance'].values
            full_energies = self.full_reference_data['Energy'].values
        else:
            # Fallback column names
            full_distances = self.full_reference_data.iloc[:, 0].values
            full_energies = self.full_reference_data.iloc[:, 1].values
        
        # Calculate number of training points
        n_total = len(full_distances)
        n_train = int(n_total * self.train_fraction)
        
        # Use random split for proper statistical validation
        # Set seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split the data
        self.train_distances = full_distances[train_indices]
        self.train_energies = full_energies[train_indices]
        self.test_distances = full_distances[test_indices]
        self.test_energies = full_energies[test_indices]
        
        # Create training reference DataFrame
        self.reference_data = pd.DataFrame({
            'Distance': self.train_distances,
            'Energy': self.train_energies
        })
        
        # Create test reference DataFrame
        self.test_reference_data = pd.DataFrame({
            'Distance': self.test_distances,
            'Energy': self.test_energies
        })
        
        logger.info(f"Data split: {n_train} training points, {len(test_indices)} test points")
        logger.info(f"Training distance range: {self.train_distances.min():.2f} - {self.train_distances.max():.2f} Å")
        logger.info(f"Test distance range: {self.test_distances.min():.2f} - {self.test_distances.max():.2f} Å")


def main():
    """Example usage with different systems"""
    import sys
    
    # Allow system selection from command line
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
    else:
        system_name = "H2"  # Default
    
    print(f"Running GA optimization for {system_name}")
    
    # System-optimized configuration
    config = GAConfig(
        population_size=30,
        generations=50,
        mutation_rate=0.2,
        crossover_rate=0.8,
        max_workers=16  # Better utilize available CPUs
    )
    
    # Initialize GA
    ga = GeneralParameterGA(
        system_name=system_name,
        base_param_file=str(BASE_PARAM_FILE),
        config=config
    )
    
    # Run optimization
    best_individual = ga.optimize()
    
    # Save results
    ga.save_best_parameters(ga.system_config.optimized_params_file)
    ga.save_fitness_history(ga.system_config.fitness_history_file)
    
    # Print best parameters
    print(f"\nBest Parameters for {system_name}:")
    for param_name, value in best_individual.parameters.items():
        print(f"  {param_name}: {value:.6f}")
    
    print(f"\nBest Fitness: {best_individual.fitness:.6f}")


if __name__ == "__main__":
    main()
