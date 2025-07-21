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
from utils.parameter_bounds import ParameterBounds

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

from calculators.calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from utils.data_extraction import GFN1ParameterExtractor
from config import get_system_config, SystemConfig

BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"


@dataclass
class CMA2Config:
    """Configuration for pycma CMA-ES optimization"""
    sigma: float = 0.1  # Initial step size 
    max_generations: int = 500  # Maximum number of generations 
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
        

    
    def evaluate_cma_fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness for pycma CMA-ES (minimizes RMSE directly)"""
        try:
            # Convert numpy array to parameter dictionary
            param_names = [bound.name for bound in self.parameter_bounds]
            parameters = {param_names[i]: float(x[i]) for i in range(len(param_names))}
            # No need to apply bounds or penalty: pycma enforces bounds natively
            rmse = self.evaluate_fitness(parameters)
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

        # Validate parameter bounds using centralized system
        validation_errors = self.bounds_manager.validate_parameters(
            {bound.name: bound.default_val for bound in self.parameter_bounds}, 
            self.parameter_bounds
        )
        if validation_errors:
            logger.error("Parameter validation errors:")
            for error in validation_errors:
                logger.error(f"  {error}")
            raise ValueError("Invalid parameter bounds detected. Check the log for details.")

        # Prepare bounds for pycma in correct format: [lower_bounds, upper_bounds]
        lower_bounds = np.array([bound.min_val for bound in self.parameter_bounds])
        upper_bounds = np.array([bound.max_val for bound in self.parameter_bounds])
        
        # Log bounds summary using centralized system
        self.bounds_manager.log_bounds_summary(self.parameter_bounds, self.system_name)

        # Format bounds for pycma: [lower_bounds, upper_bounds]
        bounds = [lower_bounds, upper_bounds]



        # Initialize CMA-ES with bounds
        self.es = cma.CMAEvolutionStrategy(
            initial_mean,
            self.config.sigma,
            {
                'maxiter': self.config.max_generations,
                'popsize': self.config.population_size,
                'seed': self.config.seed,
                'bounds': bounds,  # Pass bounds in correct format
                'CMA_diagonal': True,
                'CMA_elitist': True,
                'tolfun': 1e-6,
                'tolx': 1e-6
            }
        )
        
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


 
