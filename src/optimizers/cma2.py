"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) using pycma library
"""
import numpy as np
import pandas as pd
import logging
import os
import pickle
import multiprocessing
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

# Global variable to hold evaluator instance for multiprocessing
# This is set per worker process via pool initializer
_global_evaluator = None

class _FitnessEvaluator:
    """Concrete evaluator class for worker processes (not abstract)."""
    def __init__(self, system_name: str, base_param_file: str,
                 reference_data, train_fraction: float, spin: int,
                 param_names: List[str], parameter_bounds):
        from base_optimizer import BaseOptimizer
        
        # Store evaluation data
        self.param_names = param_names
        self.parameter_bounds = parameter_bounds
        
        # Create a concrete optimizer class that implements optimize()
        class _WorkerOptimizer(BaseOptimizer):
            def optimize(self):
                # No-op for worker processes
                return {}
        
        # Create optimizer instance manually to skip checkpoint loading
        # This avoids the recursion issue where checkpoint loading tries to set up pool
        self._optimizer = _WorkerOptimizer.__new__(_WorkerOptimizer)
        
        # Manually initialize all required attributes (mimicking BaseOptimizer.__init__)
        import toml
        from config import get_system_config, CalculationType
        from utils.parameter_bounds import ParameterBoundsManager
        
        self._optimizer.system_name = system_name
        self._optimizer.base_param_file = Path(base_param_file)
        self._optimizer.train_fraction = train_fraction
        self._optimizer.spin = spin
        self._optimizer.run_name = None
        self._optimizer.checkpoint_dir = None
        
        # Load base parameters
        with open(base_param_file, 'r') as f:
            self._optimizer.base_params = toml.load(f)
        
        # Setup system config and parameter bounds
        self._optimizer.system_config = get_system_config(system_name)
        self._optimizer.bounds_manager = ParameterBoundsManager()
        self._optimizer.parameter_bounds = parameter_bounds
        
        # Store original parameter values for delta calculation
        self._optimizer.original_parameters = {bound.name: bound.default_val for bound in parameter_bounds}
        
        # Cache structures for BULK calculations (always initialize, even if not BULK)
        # Must match BaseOptimizer.__init__ order: set cached_structures before _setup_reference_data
        self._optimizer.cached_structures = None
        if self._optimizer.system_config.calculation_type == CalculationType.BULK:
            try:
                self._optimizer._load_and_cache_structures()
            except Exception as e:
                # If loading structures fails, log warning but continue with empty list
                import logging
                worker_logger = logging.getLogger(f"{__name__}.worker")
                worker_logger.warning(f"Failed to load cached structures in worker: {e}")
                self._optimizer.cached_structures = []
        
        # Setup reference data (this calls _split_train_test_data internally)
        # This must come after cached_structures is initialized
        self._optimizer._setup_reference_data(reference_data)
        
        # Initialize state (skip checkpoint loading)
        # This sets best_parameters, best_fitness, convergence_counter, fitness_history, success_evaluations, etc.
        self._optimizer._init_optimization_state(None)
    
    def evaluate_cma_fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness for pycma CMA-ES (minimizes RMSE directly)"""
        try:
            # Convert numpy array to parameter dictionary
            parameters = {self.param_names[i]: float(x[i]) for i in range(len(self.param_names))}
            
            # Apply bounds
            parameters = self._optimizer.apply_bounds(parameters)
            
            # Evaluate fitness using base class method and convert to RMSE
            fitness = self._optimizer.evaluate_fitness(parameters)
            
            # Convert fitness to RMSE: fitness = 1/(1+rmse), so rmse = (1/fitness) - 1
            # Use large finite value instead of inf to avoid CMA-ES convergence issues
            if fitness > 0 and np.isfinite(fitness):
                rmse = (1.0 / fitness) - 1.0
                # Ensure RMSE is finite and reasonable
                if not np.isfinite(rmse) or rmse < 0:
                    rmse = 1000.0
            else:
                # Failed evaluation or invalid fitness
                rmse = 1000.0  # Large finite penalty instead of inf
            return rmse
        except Exception as e:
            # Log error in worker process (errors in workers might not show in main process logs)
            import logging
            worker_logger = logging.getLogger(f"{__name__}.worker")
            worker_logger.warning(f"Fitness evaluation failed in worker: {e}", exc_info=True)
            # Return large finite value for failed evaluations (CMA-ES handles finite values better than inf)
            return 1000.0

def _init_worker(optimizer_data):
    """Initialize worker process with optimizer data."""
    global _global_evaluator
    # Create evaluator instance (concrete, not abstract)
    _global_evaluator = _FitnessEvaluator(
        optimizer_data['system_name'],
        optimizer_data['base_param_file'],
        optimizer_data['train_reference_data'],
        optimizer_data['train_fraction'],
        optimizer_data['spin'],
        optimizer_data['param_names'],
        optimizer_data['parameter_bounds']
    )

def _evaluate_solution_wrapper(solution):
    """Picklable wrapper for solution evaluation function used with multiprocessing."""
    assert _global_evaluator is not None, "Global evaluator not set in worker process"
    return _global_evaluator.evaluate_cma_fitness(solution)

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
    max_workers: int = 12  # Number of parallel workers for multiprocessing (n_jobs is alias)
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
        self.best_fitness = 1000.0  # Initialize with large finite value (penalty value) for minimization
        
        # Use max_workers if n_jobs is set (backward compatibility)
        if self.config.n_jobs > 1 and self.config.max_workers == 12:
            self.config.max_workers = self.config.n_jobs
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin, **kwargs)
        
        # Initialize multiprocessing pool for parallel evaluation
        # Delay pool setup until optimize() is called to avoid issues during checkpoint loading
        self.pool = None
    
    def evaluate_cma_fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness for pycma CMA-ES (minimizes RMSE directly)"""
        try:
            # Convert numpy array to parameter dictionary
            param_names = [bound.name for bound in self.parameter_bounds]
            parameters = {param_names[i]: float(x[i]) for i in range(len(param_names))}
            
            # Evaluate fitness using base class method and convert to RMSE
            fitness = self.evaluate_fitness(parameters)
            
            # Convert fitness to RMSE: fitness = 1/(1+rmse), so rmse = (1/fitness) - 1
            # Use large finite value instead of inf to avoid CMA-ES convergence issues
            if fitness > 0 and np.isfinite(fitness):
                rmse = (1.0 / fitness) - 1.0
                # Ensure RMSE is finite and reasonable
                if not np.isfinite(rmse) or rmse < 0:
                    rmse = 1000.0
            else:
                # Failed evaluation or invalid fitness
                rmse = 1000.0  # Large finite penalty instead of inf
            
            return rmse  # pycma minimizes directly
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            self.failed_evaluations += 1
            # Return large finite value for failed evaluations (CMA-ES handles finite values better than inf)
            return 1000.0
    
    def _setup_pool(self):
        """Setup multiprocessing pool for parallel evaluation."""
        # Cleanup existing pool if any
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        
        # Create new pool if parallel execution is requested
        if self.config.max_workers > 1:
            try:
                # Prepare optimizer data for worker initialization
                optimizer_data = {
                    'system_name': self.system_name,
                    'base_param_file': str(self.base_param_file),
                    'train_reference_data': getattr(self, 'reference_data', None),
                    'train_fraction': self.train_fraction,
                    'spin': self.spin,
                    'param_names': [bound.name for bound in self.parameter_bounds],
                    'parameter_bounds': self.parameter_bounds
                }
                # Create pool with initializer to set up optimizer in each worker
                self.pool = multiprocessing.Pool(
                    processes=self.config.max_workers,
                    initializer=_init_worker,
                    initargs=(optimizer_data,)
                )
                logger.info(f"Initialized multiprocessing pool with {self.config.max_workers} workers for parallel fitness evaluation")
            except Exception as e:
                # Fallback to serial execution if multiprocessing fails
                logger.warning(f"Failed to create multiprocessing pool: {e}. Falling back to serial execution.")
                self.pool = None
        else:
            self.pool = None
            logger.info("Using serial execution (max_workers = 1)")
    
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

        # Ensure pool is set up (in case optimize is called directly without checkpoint)
        # Also set up pool if it wasn't created during checkpoint loading
        if not hasattr(self, 'pool') or (self.config.max_workers > 1 and self.pool is None):
            self._setup_pool()
        
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
            if hasattr(self, 'best_fitness') and self.best_fitness < 1000.0:
                logger.info(f"  Current best RMSE: {self.best_fitness:.6f}")
            elif hasattr(self, 'best_fitness') and self.best_fitness >= 1000.0:
                logger.info(f"  Current best RMSE: {self.best_fitness:.6f} (penalty - no valid evaluations yet)")
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
        
        # Run optimization with parallel fitness evaluation
        try:
            # Track fitness during optimization
            if self.generation >= self.config.max_generations:
                logger.info("Optimization already completed based on checkpoint")
                return self.best_parameters or {}

            while self.generation < self.config.max_generations:
                # Get solutions for current generation
                solutions = self.es.ask()
                
                # Evaluate fitness for all solutions (parallel or serial)
                if self.pool is not None:
                    # Parallel evaluation using multiprocessing pool
                    fitness_values = self.pool.map(_evaluate_solution_wrapper, solutions)
                    # Count failures (values >= 1000.0 indicate failures, or non-finite values)
                    failed_count = sum(1 for f in fitness_values if not np.isfinite(f) or f >= 1000.0)
                    if failed_count > 0:
                        self.failed_evaluations += failed_count
                        if failed_count == len(fitness_values):
                            logger.error(f"All {len(fitness_values)} evaluations failed in generation {self.generation + 1}")
                else:
                    # Serial evaluation
                    fitness_values = []
                    for solution in solutions:
                        fitness = self.evaluate_cma_fitness(solution)
                        fitness_values.append(fitness)
                    
                    # Check for all failures in serial mode
                    failed_count = sum(1 for f in fitness_values if not np.isfinite(f) or f >= 1000.0)
                    if failed_count == len(fitness_values):
                        logger.error(f"All {len(fitness_values)} evaluations failed in generation {self.generation + 1}")
                        # Early stop if all evaluations fail
                        break
                
                # Tell CMA-ES the results
                self.es.tell(solutions, fitness_values)
                
                # Update generation counter
                self.generation += 1
                
                # Record fitness statistics for this generation
                # Filter out failed evaluations (>= 1000.0) for statistics if not all failed
                valid_fitness = [f for f in fitness_values if np.isfinite(f) and f < 1000.0]
                if valid_fitness:
                    best_rmse_gen = min(valid_fitness)
                    avg_rmse_gen = np.mean(valid_fitness)
                    std_rmse_gen = np.std(valid_fitness)
                else:
                    # All evaluations failed
                    best_rmse_gen = min(fitness_values)  # Use minimum even if all are penalties
                    avg_rmse_gen = np.mean(fitness_values)
                    std_rmse_gen = np.std(fitness_values)
                    logger.warning(f"All evaluations returned penalty values in generation {self.generation}")
                
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
                # This works correctly since penalty value (1000.0) is much larger than valid RMSE values
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
            
            total_time = time.time() - start_time
            logger.info(f"Optimization completed in {total_time:.2f}s")
            logger.info(f"Total failed evaluations: {self.failed_evaluations}")
            logger.info(f"Best RMSE: {self.best_fitness:.6f}")
            logger.info(f"Note: Fitness history stores RMSE values (lower is better)")
            
            # Save fitness history to CSV
            self._save_fitness_history()
            
            return self.best_parameters
            
        except Exception as e:
            logger.error(f"pycma CMA-ES optimization failed: {e}")
            raise RuntimeError(f"pycma CMA-ES optimization failed: {e}") from e
        finally:
            # Cleanup multiprocessing pool in finally block to ensure it's always closed
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
    
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
        # Override base class set_state to use penalty value instead of inf for best_fitness
        self.best_parameters = state.get('best_parameters')
        # Use penalty value (1000.0) instead of inf as default
        self.best_fitness = state.get('best_fitness', 1000.0)
        # Convert inf to penalty value if present (for backward compatibility with old checkpoints)
        if self.best_fitness == float('inf') or not np.isfinite(self.best_fitness):
            self.best_fitness = 1000.0
        self.fitness_history = state.get('fitness_history', [])
        self.convergence_counter = state.get('convergence_counter', 0)
        if 'original_parameters' in state:
            self.original_parameters = state['original_parameters']
        
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
        
        # Recreate multiprocessing pool after checkpoint load (pool can't be pickled)
        # Delay pool setup until optimize() is called to avoid issues during checkpoint loading
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def __del__(self):
        """Cleanup pool on object deletion."""
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception:
                pass  # Ignore errors during cleanup
