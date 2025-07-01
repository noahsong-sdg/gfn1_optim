"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for TBLite parameter optimization - Refactored to use BaseOptimizer
"""

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import copy
from base_optimizer import BaseOptimizer

# External CMA-ES library
try:
    from cmaes import CMA
    HAS_CMAES = True
except ImportError:
    HAS_CMAES = False

# Set up logging with better control
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Control verbosity of external libraries
logging.getLogger('cmaes').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Import project modules
from calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from data_extraction import GFN1ParameterExtractor
from config import get_system_config, SystemConfig

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"


@dataclass
class CMAConfig:
    """Configuration for CMA-ES optimization"""
    sigma: float = 0.5  # Initial step size (standard deviation)
    max_generations: int = 40  # Maximum number of generations (reduced for efficiency)
    population_size: Optional[int] = None  # If None, uses CMA-ES default (4 + 3*log(dim))
    seed: Optional[int] = None  # Random seed for reproducibility
    convergence_threshold: float = 1e-6  # Fitness improvement threshold
    patience: int = 20  # Generations without improvement before stopping
    bounds_handling: str = "repair"  # How to handle parameter bounds: "repair" or "penalty"


class GeneralParameterCMA(BaseOptimizer):
    """CMA-ES optimizer inheriting from BaseOptimizer"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: CMAConfig = CMAConfig(),
                 train_fraction: float = 0.8):
        """Initialize CMA-ES optimizer"""
        
        if not HAS_CMAES:
            raise ImportError("cmaes library is required for CMA-ES optimization. "
                            "Install with: pip install cmaes")
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction)
        
        # CMA-ES specific configuration
        self.config = config
        
        # CMA-ES specific state
        self.optimizer = None
        self.generation = 0
        self.failed_evaluations = 0
        
    def apply_bounds(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply parameter bounds by clamping values"""
        bounded_params = {}
        for param_name, value in parameters.items():
            bound = next((b for b in self.parameter_bounds if b.name == param_name), None)
            if bound:
                bounded_value = max(bound.min_val, min(bound.max_val, value))
                
                # Extra safety check for Slater exponents
                if 'slater' in param_name:
                    bounded_value = max(0.5, bounded_value)  # Absolute minimum for safety
                
                bounded_params[param_name] = float(bounded_value)
            else:
                bounded_params[param_name] = float(value)
        return bounded_params
    
    def evaluate_cma_fitness(self, parameters: Dict[str, float]) -> float:
        """Evaluate fitness for CMA-ES (minimizes RMSE directly)"""
        try:
            # Apply bounds if using repair method
            if self.config.bounds_handling == "repair":
                parameters = self.apply_bounds(parameters)
            
            # Evaluate fitness using base class method
            rmse = self.evaluate_fitness(parameters)
            
            # Apply penalty for out-of-bounds parameters if using penalty method
            if self.config.bounds_handling == "penalty":
                penalty = 0.0
                for param_name, value in parameters.items():
                    bound = next((b for b in self.parameter_bounds if b.name == param_name), None)
                    if bound and (value < bound.min_val or value > bound.max_val):
                        penalty += 1000.0  # Large penalty for constraint violation
                rmse += penalty
            
            return rmse  # CMA-ES minimizes directly
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            self.failed_evaluations += 1
            return float('inf')  # Return very large value for failed evaluations
    
    def check_convergence(self) -> bool:
        """Check if the algorithm has converged"""
        if len(self.fitness_history) < 2:
            return False
        
        # Check if improvement is below threshold
        recent_improvement = abs(
            self.fitness_history[-2]['best_fitness'] - 
            self.fitness_history[-1]['best_fitness']
        )
        
        if recent_improvement < self.config.convergence_threshold:
            return self.convergence_counter >= self.config.patience
        
        return False
    
    def optimize(self) -> Dict[str, float]:
        """Run CMA-ES optimization"""
        logger.info(f"Starting CMA-ES optimization for {self.system_name}")
        start_time = time.time()
        
        # Prepare initial parameters
        param_names = [bound.name for bound in self.parameter_bounds]
        initial_mean = np.array([bound.default_val for bound in self.parameter_bounds])
        
        logger.info(f"Parameter summary:")
        for i, bound in enumerate(self.parameter_bounds[:5]):  # Show first 5 parameters
            logger.info(f"  {bound.name}: default={bound.default_val:.4f}, range=[{bound.min_val:.4f}, {bound.max_val:.4f}]")
        if len(param_names) > 5:
            logger.info(f"  ... and {len(param_names)-5} more parameters")
        
        # Initialize CMA-ES optimizer
        try:
            self.optimizer = CMA(
                mean=initial_mean,
                sigma=self.config.sigma,
                population_size=self.config.population_size,
                seed=self.config.seed
            )
            logger.info(f"CMA-ES initialized with population size: {self.optimizer.population_size}")
        except Exception as e:
            logger.error(f"Failed to initialize CMA-ES optimizer: {e}")
            raise RuntimeError(f"CMA-ES initialization failed: {e}") from e
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Log every 10 generations or first/last few
            should_log = (generation % 10 == 0 or generation < 3 or 
                         generation >= self.config.max_generations - 3)
            
            if should_log:
                logger.info(f"Generation {generation + 1}/{self.config.max_generations}")
            
            # Ask for new parameter sets
            solutions = []
            gen_start = time.time()
            
            for _ in range(self.optimizer.population_size):
                # Get parameter vector from CMA-ES
                x = self.optimizer.ask()
                
                # Convert to parameter dictionary (ensure numpy types become Python floats)
                params = {param_names[i]: float(x[i]) for i in range(len(param_names))}
                
                # Evaluate fitness
                fitness = self.evaluate_cma_fitness(params)
                solutions.append((x, fitness))
            
            # Tell CMA-ES the results
            self.optimizer.tell(solutions)
            
            eval_time = time.time() - gen_start
            
            # Track best solution
            best_solution = min(solutions, key=lambda x: x[1])
            best_x, best_fitness = best_solution
            best_params = {param_names[i]: float(best_x[i]) for i in range(len(param_names))}
            
            # Update best if improved
            if self.best_parameters is None or best_fitness < self.best_fitness:
                self.best_parameters = best_params.copy()
                self.best_fitness = best_fitness
                self.convergence_counter = 0
                logger.info(f"  NEW BEST at gen {generation + 1}: RMSE = {best_fitness:.6f}")
            else:
                self.convergence_counter += 1
            
            # Record fitness history
            avg_fitness = np.mean([sol[1] for sol in solutions])
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std([sol[1] for sol in solutions])
            })
            
            # Log progress (reduced frequency)
            if should_log:
                logger.info(f"  Best RMSE: {best_fitness:.6f} | Avg: {avg_fitness:.6f} | "
                           f"Fails: {self.failed_evaluations} | Time: {eval_time:.1f}s")
            
            # Early stopping if all fitness values are inf
            if best_fitness == float('inf'):
                logger.error("All individuals have infinite fitness - stopping optimization")
                break
            
            # Check convergence
            if self.check_convergence():
                logger.info(f"Converged at generation {generation + 1}")
                break
            
            # Check CMA-ES stopping condition
            if self.optimizer.should_stop():
                logger.info(f"CMA-ES stopping criterion met at generation {generation + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        logger.info(f"Total failed evaluations: {self.failed_evaluations}")
        
        return self.best_parameters


def main():
    """Example usage with different systems"""
    import sys
    from pathlib import Path
    
    if not HAS_CMAES:
        print("Error: cmaes library is required for CMA-ES optimization")
        print("Install with: pip install cmaes")
        sys.exit(1)
    
    PROJECT_ROOT = Path.cwd()
    CONFIG_DIR = PROJECT_ROOT / "config"
    BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
    
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
    else:
        system_name = "H2"
    
    print(f"Running CMA-ES optimization for {system_name}")
    
    config = CMAConfig(sigma=0.5, max_generations=40)
    cma = GeneralParameterCMA(system_name, str(BASE_PARAM_FILE), config=config)
    best_parameters = cma.optimize()
    
    # Save results using base class methods
    cma.save_best_parameters(cma.system_config.optimized_params_file)
    cma.save_fitness_history(cma.system_config.fitness_history_file)
    
    if best_parameters:
        print(f"\nBest Parameters for {system_name}:")
        for param_name, value in best_parameters.items():
            print(f"  {param_name}: {value:.6f}")


if __name__ == "__main__":
    main() 
