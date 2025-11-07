"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for TBLite parameter optimization - Refactored to use BaseOptimizer

ONE SHOT
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import time
from base_optimizer import BaseOptimizer
from cmaes import CMA

# Set up logging with better control
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# shut up
logging.getLogger('cmaes').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

@dataclass
class CMAConfig:
    """Configuration for CMA-ES optimization"""
    sigma: float = 0.1  
    max_generations: int = 100  
    population_size: Optional[int] = None  # If None, uses CMA-ES default (4 + 3*log(dim))
    seed: Optional[int] = None  
    convergence_threshold: float = 1e-6  # Fitness improvement threshold
    patience: int = 20  # 
    bounds_handling: str = "penalty"  #  "repair" or "penalty"

class GeneralParameterCMA(BaseOptimizer):
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: CMAConfig = CMAConfig(),
                 train_fraction: float = 0.8,
                 spin: int = 0,
                 **kwargs):
        self.config = config
        self.optimizer = None
        self.generation = 0
        self.failed_evaluations = 0
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin, **kwargs)
    
    def evaluate_cma_fitness(self, parameters: Dict[str, float]) -> float:
        """Evaluate fitness for CMA-ES (minimizes RMSE directly)"""
        try:
            # Apply bounds if using repair method
            if self.config.bounds_handling == "repair":
                parameters = self.apply_bounds(parameters)
            
            # Evaluate fitness using base class method and convert to RMSE algebraically using the inverse transformation.
            fitness = self.evaluate_fitness(parameters)
            rmse = (1.0 / fitness) - 1.0 if fitness > 0 else float('inf')
            
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
            
            # Calculate parameter deltas for best solution
            param_deltas = self.calculate_parameter_deltas(best_params)
            
            # Record fitness history with parameter deltas
            avg_fitness = np.mean([sol[1] for sol in solutions])
            history_entry = {
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std([sol[1] for sol in solutions])
            }
            # Add parameter deltas with 'delta_' prefix
            for param_name, delta in param_deltas.items():
                history_entry[f'delta_{param_name}'] = delta
            
            self.fitness_history.append(history_entry)
            
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
    
    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            'failed_evaluations': self.failed_evaluations,
            'generation': self.generation,
            'config': self.config
        })
        # Add CMA-ES specific state if optimizer exists
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            state['cma_state'] = {
                'mean': self.optimizer.mean.copy(),
                'sigma': self.optimizer.sigma,
                'population_size': self.optimizer.population_size,
                'generation': self.optimizer.generation
            }
        return state

    def set_state(self, state: dict):
        super().set_state(state)
        self.failed_evaluations = state.get('failed_evaluations', 0)
        self.generation = state.get('generation', 0)
        self.config = state.get('config', self.config)
        # Restore CMA-ES state if available
        if 'cma_state' in state and hasattr(self, 'optimizer') and self.optimizer is not None:
            cma_state = state['cma_state']
            self.optimizer.mean = cma_state['mean']
            self.optimizer.sigma = cma_state['sigma']
            self.optimizer.population_size = cma_state['population_size']
            self.optimizer.generation = cma_state['generation']
