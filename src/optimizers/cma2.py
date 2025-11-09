"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) using pycma library
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
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

@dataclass
class CMA2Config:
    """Configuration for pycma CMA-ES optimization"""
    sigma: float = 0.1  
    max_generations: int = 12  
    population_size: Optional[int] = None  # none means uses CMA-ES default
    seed: Optional[int] = None  
    convergence_threshold: float = 1e-6  
    patience: int = 20  
    n_jobs: int = 12  # Number of parallel jobs (1 = sequential)
    verb_disp: int = 1  # Verbosity level 0, 1, 2
    tolfun: float = 1e-6  # Function value tolerance
    tolx: float = 1e-6  # Solution tolerance
    maxiter: int = 150  # Maximum iterations
    bounds_handling: str = "penalty"  # "repair" or "penalty"

class GeneralParameterCMA2(BaseOptimizer):
    """pycma CMA-ES optimizer inheriting from BaseOptimizer"""
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: CMA2Config = CMA2Config(),
                 train_fraction: float = 0.8,
                 spin: int = 0,
                 **kwargs):
        """Initialize pycma CMA-ES optimizer"""
        self.config = config
        
        # pycma specific state
        self.es = None  # CMAEvolutionStrategy instance
        self.generation = 0
        self.failed_evaluations = 0
        self.best_fitness = float('inf')  # Initialize for minimization
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin, **kwargs)
    
    def evaluate_cma_fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness for pycma CMA-ES (minimizes RMSE directly)"""
        try:
            # Convert numpy array to parameter dictionary
            param_names = [bound.name for bound in self.parameter_bounds]
            parameters = {param_names[i]: float(x[i]) for i in range(len(param_names))}
            
            # Evaluate fitness using base class method and convert to RMSE
            fitness = self.evaluate_fitness(parameters)
            rmse = (1.0 / fitness) - 1.0 if fitness > 0 else float('inf')
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
        """Run pycma CMA-ES optimization with checkpoint/resume support"""
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

        # Initialize or resume CMA-ES
        # Check if we're resuming from a checkpoint (either es is restored or generation > 0)
        resumed = (self.es is not None) or (self.generation > 0)
        if resumed:
            completed = self.generation
            remaining = max(0, self.config.max_generations - self.generation)
            progress_pct = (completed / self.config.max_generations * 100) if self.config.max_generations > 0 else 0
            logger.info(f"Resuming CMA-ES optimization from checkpoint")
            logger.info(f"  Progress: {completed}/{self.config.max_generations} generations completed ({progress_pct:.1f}%)")
            logger.info(f"  Remaining: {remaining} generations to complete")
            if hasattr(self, 'best_fitness') and self.best_fitness != float('inf'):
                logger.info(f"  Current best RMSE: {self.best_fitness:.6f}")
        else:
            logger.info("Starting fresh CMA-ES optimization")
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
            # Track fitness during optimization
            if self.generation >= self.config.max_generations:
                logger.info("Optimization already completed based on checkpoint")
                return self.best_parameters or {}

            while self.generation < self.config.max_generations:
                # Get solutions for current generation
                solutions = self.es.ask()
                
                # Evaluate fitness for all solutions
                fitness_values = []
                for solution in solutions:
                    fitness = self.evaluate_cma_fitness(solution)
                    fitness_values.append(fitness)
                
                # Tell CMA-ES the results
                self.es.tell(solutions, fitness_values)
                
                # Update generation counter
                self.generation += 1
                
                # Record fitness statistics for this generation
                best_rmse_gen = min(fitness_values)  # CMA-ES minimizes RMSE
                avg_rmse_gen = np.mean(fitness_values)
                std_rmse_gen = np.std(fitness_values)
                
                # Get best solution parameters for delta calculation
                best_idx = np.argmin(fitness_values)
                best_solution = solutions[best_idx]
                best_params_gen = {param_names[i]: float(best_solution[i]) for i in range(len(param_names))}
                
                # Calculate parameter deltas
                param_deltas = self.calculate_parameter_deltas(best_params_gen)
                
                # Create history entry with fitness and parameter deltas
                history_entry = {
                    'generation': self.generation,
                    'best_fitness': best_rmse_gen,  # Store RMSE (lower is better)
                    'avg_fitness': avg_rmse_gen,
                    'std_fitness': std_rmse_gen
                }
                # Add parameter deltas with 'delta_' prefix
                for param_name, delta in param_deltas.items():
                    history_entry[f'delta_{param_name}'] = delta
                
                self.fitness_history.append(history_entry)
                
                # Update best parameters if we have a better solution (lower RMSE is better)
                if best_rmse_gen < self.best_fitness:
                    self.best_parameters = best_params_gen.copy()
                    self.best_fitness = best_rmse_gen
                    # Save checkpoint on improvement
                    self.save_checkpoint()
                
                # Log progress
                if self.generation % 5 == 0 or self.generation == 1:
                    logger.info(f"Generation {self.generation}: Best RMSE = {best_rmse_gen:.6f}, Avg = {avg_rmse_gen:.6f}")
                    # Periodic checkpoint
                    self.save_checkpoint()
                
                # Check convergence
                if self.es.stop():
                    logger.info(f"CMA-ES converged at generation {self.generation}")
                    self.save_checkpoint()
                    break
            
            # Extract final best solution
            best_x = self.es.result.xbest
            best_fitness = self.es.result.fbest
            
            # Convert to parameter dictionary
            best_params = {param_names[i]: float(best_x[i]) for i in range(len(param_names))}
            
            # Update best parameters
            self.best_parameters = best_params.copy()
            self.best_fitness = best_fitness
            # Final checkpoint at completion
            self.save_checkpoint()
            
        except Exception as e:
            logger.error(f"pycma CMA-ES optimization failed: {e}")
            raise RuntimeError(f"pycma CMA-ES optimization failed: {e}") from e
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        logger.info(f"Total failed evaluations: {self.failed_evaluations}")
        logger.info(f"Best RMSE: {self.best_fitness:.6f}")
        logger.info(f"Note: Fitness history stores RMSE values (lower is better)")
        
        # Save fitness history to CSV
        self._save_fitness_history()
        
        return self.best_parameters
    
    def _save_fitness_history(self):
        """Save fitness history to CSV file"""
        if not self.fitness_history:
            logger.warning("No fitness history to save")
            return
        
        try:
            # Create results directory if it doesn't exist
            fitness_dir = RESULTS_DIR / "fitness"
            fitness_dir.mkdir(parents=True, exist_ok=True)
            
            # Save fitness history
            fitness_file = fitness_dir / f"{self.system_name}_cma2_history.csv"
            df = pd.DataFrame(self.fitness_history)
            df.to_csv(fitness_file, index=False)
            logger.info(f"Fitness history saved to {fitness_file}")
            
        except Exception as e:
            logger.error(f"Failed to save fitness history: {e}")
    
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
