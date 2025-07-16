"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) using pycma library
Refactored to use BaseOptimizer with official pycma implementation
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import copy
from base_optimizer import BaseOptimizer

import cma

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# shut up shut up shut up
logging.getLogger('cma').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

from calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from data_extraction import GFN1ParameterExtractor
from config import get_system_config, SystemConfig

BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"


@dataclass
class CMA2Config:
    """Configuration for pycma CMA-ES optimization"""
    sigma: float = 0.1  # Initial step size 
    max_generations: int = 50  # Maximum number of generations 
    population_size: Optional[int] = None  # If None, uses CMA-ES default
    seed: Optional[int] = None  
    convergence_threshold: float = 1e-6  # Fitness improvement threshold
    patience: int = 20  # Generations without improvement before stopping
    n_jobs: int = 1  # Number of parallel jobs (1 = sequential)
    verb_disp: int = 1  # Verbosity level (0 = silent, 1 = progress, 2 = detailed)
    tolfun: float = 1e-6  # Function value tolerance
    tolx: float = 1e-6  # Solution tolerance
    maxiter: int = 1000  # Maximum iterations
    bounds_handling: str = "penalty"  # "repair" or "penalty"


class GeneralParameterCMA2(BaseOptimizer):
    """pycma CMA-ES optimizer inheriting from BaseOptimizer"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: CMA2Config = CMA2Config(),
                 train_fraction: float = 0.8):
        """Initialize pycma CMA-ES optimizer"""
        
        #if not HAS_PYCMA:
        #    raise ImportError("pycma library is required for CMA-ES optimization. "
        #                    "Install with: pip install cma")
        
        # pycma specific configuration (set before super().__init__ to avoid set_state issues)
        self.config = config
        
        # pycma specific state
        self.es = None  # CMAEvolutionStrategy instance
        self.generation = 0
        self.failed_evaluations = 0
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction)
        
    def apply_bounds(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply parameter bounds by clamping values"""
        bounded_params = {}
        for param_name, value in parameters.items():
            bound = next((b for b in self.parameter_bounds if b.name == param_name), None)
            if bound:
                bounded_value = max(bound.min_val, min(bound.max_val, value))
                
                # Extra safety check for Slater exponents and other critical parameters
                if 'slater' in param_name:
                    bounded_value = max(0.5, bounded_value)  # Absolute minimum for safety
                elif 'kpair' in param_name:
                    bounded_value = max(0.1, bounded_value)  # Pair parameters must be positive
                elif 'kcn' in param_name:
                    bounded_value = max(0.01, bounded_value)  # Coordination numbers must be positive
                elif 'gam' in param_name and not param_name.endswith('lgam'):
                    bounded_value = max(0.1, min(1.0, bounded_value))  # Gamma parameters reasonable range
                elif 'zeff' in param_name:
                    bounded_value = max(1.0, bounded_value)  # Effective charge must be positive
                elif 'arep' in param_name:
                    bounded_value = max(0.5, bounded_value)  # Repulsion parameters must be positive
                elif 'en' in param_name and not param_name.endswith('zen'):
                    bounded_value = max(0.5, bounded_value)  # Electronegativity must be positive
                
                bounded_params[param_name] = float(bounded_value)
            else:
                bounded_params[param_name] = float(value)
        return bounded_params
    
    def evaluate_cma_fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness for pycma CMA-ES (minimizes RMSE directly)"""
        try:
            # Convert numpy array to parameter dictionary
            param_names = [bound.name for bound in self.parameter_bounds]
            parameters = {param_names[i]: float(x[i]) for i in range(len(param_names))}
            
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
            
            return rmse  # pycma minimizes directly
            
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
        """Run pycma CMA-ES optimization"""
        logger.info(f"Starting pycma CMA-ES optimization for {self.system_name}")
        start_time = time.time()
        
        # Prepare initial parameters
        param_names = [bound.name for bound in self.parameter_bounds]
        initial_mean = np.array([bound.default_val for bound in self.parameter_bounds])
        
        logger.info(f"Parameter summary:")
        for i, bound in enumerate(self.parameter_bounds[:5]):  # Show first 5 parameters
            logger.info(f"  {bound.name}: default={bound.default_val:.4f}, range=[{bound.min_val:.4f}, {bound.max_val:.4f}]")
        if len(param_names) > 5:
            logger.info(f"  ... and {len(param_names)-5} more parameters")
        
        # Set up pycma options
        options = {
            'maxiter': self.config.max_generations,
            'popsize': self.config.population_size,
            'seed': self.config.seed,
            'verb_disp': self.config.verb_disp,
            'tolfun': self.config.tolfun,
            'tolx': self.config.tolx,
            'CMA_diagonal': False,  # Use full covariance matrix
            'CMA_elitist': False,   # Non-elitist selection
        }
        
        # Initialize pycma CMA-ES optimizer
        try:
            self.es = cma.CMAEvolutionStrategy(
                initial_mean, 
                self.config.sigma, 
                options
            )
            logger.info(f"pycma CMA-ES initialized with population size: {self.es.popsize}")
            if self.config.n_jobs > 1:
                logger.info(f"Parallel evaluation enabled with {self.config.n_jobs} jobs")
        except Exception as e:
            logger.error(f"Failed to initialize pycma CMA-ES optimizer: {e}")
            raise RuntimeError(f"pycma CMA-ES initialization failed: {e}") from e
        
        # Run optimization with pycma's built-in parallelization
        try:
            result = self.es.optimize(
                self.evaluate_cma_fitness,
                iterations=self.config.max_generations,
                n_jobs=self.config.n_jobs
            )
            
            # Extract best solution
            best_x = self.es.result.xbest
            best_fitness = self.es.result.fbest
            
            # Convert to parameter dictionary
            best_params = {param_names[i]: float(best_x[i]) for i in range(len(param_names))}
            
            # Update best parameters
            self.best_parameters = best_params.copy()
            self.best_fitness = best_fitness
            
            # Record final statistics
            self.fitness_history.append({
                'generation': self.es.generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(self.es.fit.fit),
                'std_fitness': np.std(self.es.fit.fit)
            })
            
        except Exception as e:
            logger.error(f"pycma CMA-ES optimization failed: {e}")
            raise RuntimeError(f"pycma CMA-ES optimization failed: {e}") from e
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        logger.info(f"Total failed evaluations: {self.failed_evaluations}")
        logger.info(f"Best RMSE: {self.best_fitness:.6f}")
        
        return self.best_parameters
    
    def get_state(self) -> dict:
        """Get checkpoint state including pycma-specific state"""
        state = super().get_state()
        state.update({
            'failed_evaluations': self.failed_evaluations,
            'generation': self.generation,
            'config': self.config
        })
        
        # Add pycma-specific state if optimizer exists
        if hasattr(self, 'es') and self.es is not None:
            # Save pycma state using pickle
            state['pycma_state'] = pickle.dumps(self.es)
        
        return state

    def set_state(self, state: dict):
        """Set checkpoint state including pycma-specific state"""
        super().set_state(state)
        self.failed_evaluations = state.get('failed_evaluations', 0)
        self.generation = state.get('generation', 0)
        self.config = state.get('config', self.config)
        
        # Restore pycma state if available
        if 'pycma_state' in state and hasattr(self, 'es'):
            try:
                self.es = pickle.loads(state['pycma_state'])
                logger.info("Restored pycma CMA-ES state from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to restore pycma state: {e}")
                self.es = None


def main():
    import sys
    from pathlib import Path
    
    PROJECT_ROOT = Path.cwd()
    CONFIG_DIR = PROJECT_ROOT / "config"
    BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
    
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
    
    
    config = CMA2Config()
    
    cma2 = GeneralParameterCMA2(system_name, str(BASE_PARAM_FILE), config=config)
    best_parameters = cma2.optimize()
    
    # Save results using method-specific filenames
    cma2.save_best_parameters()  # Will use si2_cma2.toml instead of si2_optimized.toml
    cma2.save_fitness_history()  # Will use si2_cma2_fitness_history.csv
    
    if best_parameters:
        print(f"\nBest Parameters for {system_name}:")
        for param_name, value in best_parameters.items():
            print(f"  {param_name}: {value:.6f}")


if __name__ == "__main__":
    main() 
