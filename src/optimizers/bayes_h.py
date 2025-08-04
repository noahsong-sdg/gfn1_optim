"""
Bayesian Optimization for TBLite parameter optimization - Refactored to use BaseOptimizer
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import pickle
# External Bayesian optimization library
try:
    from skopt import gp_minimize, dump, load
    from skopt.space import Real
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

from base_optimizer import BaseOptimizer
from utils.parameter_bounds import ParameterBounds

logger = logging.getLogger(__name__)

# Set debug logging
logging.basicConfig(level=logging.DEBUG)

@dataclass
class BayesianConfig:
    """Configuration for Bayesian optimization"""
    n_calls: int = 200  # Number of function evaluations (increased for better sampling)
    n_initial_points: int = 20  # Number of random points to start with (increased)
    acq_func: str = "EI"  # Acquisition function: "EI", "LCB", "PI"
    acq_optimizer: str = "auto"  # Acquisition optimizer
    xi: float = 0.1  # Exploration-exploitation trade-off
    kappa: float = 1.96  # Lower confidence bound parameter
    n_restarts_optimizer: int = 5  # Number of restarts for acquisition optimization
    noise: float = 1e-10  # Noise level for Gaussian process
    random_state: Optional[int] = None  # Random state for reproducibility


class GeneralParameterBayesian(BaseOptimizer):
    """Bayesian Optimization optimizer inheriting from BaseOptimizer"""
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: BayesianConfig = BayesianConfig(),
                 train_fraction: float = 0.8,
                 spin: int = 0):
        """Initialize Bayesian optimizer"""
        
        # Bayesian-specific configuration (set before super().__init__ to avoid set_state issues)
        self.config = config
        
        # Bayesian-specific state
        self.optimization_result = None
        self.call_count = 0
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin)
        
        # Set up optimization space (after base init to access parameter_bounds)
        self.dimensions = self._create_search_space()
        
        # Use saved dimension names from checkpoint if available, otherwise create new ones
        if hasattr(self, '_saved_dimension_names') and self._saved_dimension_names is not None:
            self.dimension_names = self._saved_dimension_names
            logger.debug("Using saved dimension names from checkpoint")
        else:
            self.dimension_names = [bound.name for bound in self.parameter_bounds]
            logger.debug("Creating new dimension names")
        
    def _create_search_space(self) -> List[Real]:
        """Create the search space for Bayesian optimization"""
        dimensions = []
        logger.debug(f"Creating search space for {len(self.parameter_bounds)} parameters")
        
        # Validate bounds using centralized system
        validation_errors = self.bounds_manager.validate_parameters(
            {bound.name: bound.default_val for bound in self.parameter_bounds}, 
            self.parameter_bounds
        )
        if validation_errors:
            logger.error("Parameter validation errors:")
            for error in validation_errors:
                logger.error(f"  {error}")
            raise ValueError("Invalid parameter bounds detected. Check the log for details.")
        
        for i, bound in enumerate(self.parameter_bounds):
            logger.debug(f"  Parameter {i+1}/{len(self.parameter_bounds)}: {bound.name} = [{bound.min_val}, {bound.max_val}]")
            dimensions.append(Real(bound.min_val, bound.max_val, name=bound.name))
        
        logger.info(f"Created search space with {len(dimensions)} dimensions")
        return dimensions
    
    def objective_function(self, x: List[float]) -> float:
        """Objective function for Bayesian optimization (minimizes RMSE)"""
        self.call_count += 1
        
        # Convert parameter vector to dictionary
        parameters = {name: value for name, value in zip(self.dimension_names, x)}
        
        # Evaluate fitness using base class method and convert to RMSE
        fitness = self.evaluate_fitness(parameters)
        rmse = (1.0 / fitness) - 1.0 if fitness > 0 else float('inf')
        
        # Handle failed evaluations (infinity values) by returning a large finite penalty
        if not np.isfinite(rmse):
            rmse = 1000.0  # Large penalty for failed evaluations
            logger.warning(f"Call {self.call_count}: Evaluation failed, using penalty value {rmse}")
        
        # Record in fitness history
        self.fitness_history.append({
            'generation': self.call_count - 1,
            'best_fitness': rmse,
            'avg_fitness': rmse,  # Same as best for single point evaluation
            'std_fitness': 0.0
        })
        
        if self.call_count % 10 == 0 or rmse < self.best_fitness:
            logger.info(f"Call {self.call_count}/{self.config.n_calls}: RMSE = {rmse:.6f}")
        
        # Update best if improved (lower RMSE is better)
        if rmse < self.best_fitness:
            self.best_fitness = rmse
            self.best_parameters = parameters.copy()
            logger.info(f"  NEW BEST at call {self.call_count}: RMSE = {rmse:.6f}")
        
        # Save checkpoint every 10 calls or when we find a new best
        if self.call_count % 10 == 0 or rmse < self.best_fitness:
            self.save_checkpoint()
        
        return rmse  # Bayesian optimization minimizes
    
    def optimize(self) -> Dict[str, float]:
        """Run Bayesian optimization"""
        logger.info(f"Starting Bayesian optimization for {self.system_name}")
        logger.info(f"Using {self.config.n_calls} function evaluations with {self.config.n_initial_points} initial points")
        start_time = time.time()
        
        # Try to load checkpoint if it exists
        resumed = False
        try:
            self.load_checkpoint()
            if self.call_count > 0:  # Only consider resumed if we actually loaded some state
                logger.info(f"Resuming from checkpoint at call {self.call_count}")
                resumed = True
            else:
                logger.info("Checkpoint loaded but no progress found, starting fresh")
                resumed = False
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            logger.info("No valid checkpoint found, starting fresh optimization")
            resumed = False
        
        # Set up the decorated objective function
        @use_named_args(self.dimensions)
        def objective(**params):
            return self.objective_function([params[name] for name in self.dimension_names])
        
        try:
            # Calculate remaining calls
            remaining_calls = max(0, self.config.n_calls - self.call_count)
            
            if remaining_calls <= 0:
                logger.info("Optimization already completed based on checkpoint")
                return self.best_parameters
            
            logger.info(f"Running {remaining_calls} remaining function evaluations")
            
            # For resuming, we need to provide x0 (initial points) instead of n_initial_points
            if resumed:
                n_initial_points = 0
                # We need to provide x0 for resumed optimization
                # For now, just use the best parameters as a starting point
                if self.best_parameters:
                    x0 = [[self.best_parameters[name] for name in self.dimension_names]]
                else:
                    x0 = None
            else:
                n_initial_points = self.config.n_initial_points
                x0 = None
            
            # Run Bayesian optimization with remaining calls
            self.optimization_result = gp_minimize(
                func=objective,
                dimensions=self.dimensions,
                n_calls=remaining_calls,
                n_initial_points=n_initial_points,
                x0=x0,  # Add initial points for resumed optimization
                acq_func=self.config.acq_func,
                acq_optimizer=self.config.acq_optimizer,
                xi=self.config.xi,
                kappa=self.config.kappa,
                n_restarts_optimizer=self.config.n_restarts_optimizer,
                noise=self.config.noise,
                random_state=self.config.random_state,
                verbose=False  # We handle our own logging
            )
            
            # Extract best parameters
            best_x = self.optimization_result.x
            best_rmse = self.optimization_result.fun
            
            # Update best parameters from optimization result
            self.best_parameters = {name: value for name, value in zip(self.dimension_names, best_x)}
            self.best_fitness = best_rmse
            
            total_time = time.time() - start_time
            logger.info(f"Optimization completed in {total_time:.2f}s")
            logger.info(f"Best RMSE: {best_rmse:.6f}")
            
            # Save final checkpoint
            self.save_checkpoint()
            
            return self.best_parameters
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            raise
    
    def save_optimization_result(self, filename: str):
        """Save the optimization result for later analysis"""
        if self.optimization_result is None:
            raise ValueError("No optimization has been run")
        
        dump(self.optimization_result, filename)
        logger.info(f"Optimization result saved to {filename}")
    
    def load_optimization_result(self, filename: str):
        """Load a previous optimization result"""
        self.optimization_result = load(filename)
        logger.info(f"Optimization result loaded from {filename}")
    
    def get_convergence_data(self) -> Dict[str, List[float]]:
        """Get convergence data from the optimization result"""
        if self.optimization_result is None:
            raise ValueError("No optimization has been run")
        
        # Extract function values (convergence curve)
        func_vals = self.optimization_result.func_vals
        
        # Calculate running minimum
        running_min = []
        current_min = float('inf')
        for val in func_vals:
            if val < current_min:
                current_min = val
            running_min.append(current_min)
        
        return {
            'func_vals': func_vals.tolist(),
            'running_min': running_min,
            'x_iters': [list(x) for x in self.optimization_result.x_iters]
        }
    
    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            'call_count': self.call_count,
            'config': self.config,
            'dimension_names': self.dimension_names
        })
        
        # Exclude optimization_result from checkpointing as it contains unpicklable functions
        # The optimization_result will be regenerated when resuming
        if 'optimization_result' in state:
            del state['optimization_result']
            logger.debug("Excluded optimization_result from checkpoint to avoid pickling issues")
        
        # Debug: check for any function references in the state
        def check_for_functions(obj, path=""):
            if callable(obj):
                logger.warning(f"Found callable object at {path}: {type(obj).__name__}")
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    check_for_functions(value, f"{path}.{key}")
            elif isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    check_for_functions(value, f"{path}[{i}]")
        
        check_for_functions(state, "state")
        return state

    def set_state(self, state: dict):
        super().set_state(state)
        self.call_count = state.get('call_count', 0)
        self.config = state.get('config', self.config)
        
        # Handle dimension_names carefully - it might not exist during initialization
        if hasattr(self, 'dimension_names'):
            self.dimension_names = state.get('dimension_names', self.dimension_names)
        else:
            # If dimension_names doesn't exist yet, store it for later use
            self._saved_dimension_names = state.get('dimension_names', None)
        
        # Reset optimization_result as it's not included in checkpoints
        self.optimization_result = None
        logger.debug("Reset optimization_result during state restoration")



