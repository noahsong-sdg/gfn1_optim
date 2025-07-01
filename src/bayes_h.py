"""
Bayesian Optimization for TBLite parameter optimization - Refactored to use BaseOptimizer
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

# External Bayesian optimization library
try:
    from skopt import gp_minimize, dump, load
    from skopt.space import Real
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

from base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

@dataclass
class BayesianConfig:
    """Configuration for Bayesian optimization"""
    n_calls: int = 200  # Number of function evaluations (increased for better sampling)
    n_initial_points: int = 20  # Number of random points to start with (increased)
    acq_func: str = "EI"  # Acquisition function: "EI", "LCB", "PI"
    acq_optimizer: str = "auto"  # Acquisition optimizer
    xi: float = 0.01  # Exploration-exploitation trade-off
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
                 train_fraction: float = 0.8):
        """Initialize Bayesian optimizer"""
        
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize is required for Bayesian optimization. "
                            "Install with: pip install scikit-optimize")
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction)
        
        # Bayesian-specific configuration
        self.config = config
        
        # Bayesian-specific state
        self.optimization_result = None
        self.call_count = 0
        
        # Set up optimization space
        self.dimensions = self._create_search_space()
        self.dimension_names = [bound.name for bound in self.parameter_bounds]
        
    def _create_search_space(self) -> List[Real]:
        """Create the search space for Bayesian optimization"""
        dimensions = []
        for bound in self.parameter_bounds:
            # Ensure valid bounds
            if bound.max_val <= bound.min_val:
                logger.warning(f"Invalid bounds for {bound.name}: using default ± 10%")
                min_val = bound.default_val * 0.9
                max_val = bound.default_val * 1.1
            else:
                min_val = bound.min_val
                max_val = bound.max_val
            
            dimensions.append(Real(min_val, max_val, name=bound.name))
        
        logger.info(f"Created search space with {len(dimensions)} dimensions")
        return dimensions
    
    def objective_function(self, x: List[float]) -> float:
        """Objective function for Bayesian optimization (minimizes RMSE)"""
        self.call_count += 1
        
        # Convert parameter vector to dictionary
        parameters = {name: value for name, value in zip(self.dimension_names, x)}
        
        # Evaluate fitness using base class method
        rmse = self.evaluate_fitness(parameters)
        
        # Record in fitness history
        self.fitness_history.append({
            'generation': self.call_count - 1,
            'best_fitness': rmse,
            'avg_fitness': rmse,  # Same as best for single point evaluation
            'std_fitness': 0.0
        })
        
        if self.call_count % 10 == 0 or rmse < self.best_fitness:
            logger.info(f"Call {self.call_count}/{self.config.n_calls}: RMSE = {rmse:.6f}")
        
        # Update best if improved
        if rmse < self.best_fitness:
            self.best_fitness = rmse
            self.best_parameters = parameters.copy()
            logger.info(f"  NEW BEST at call {self.call_count}: RMSE = {rmse:.6f}")
        
        return rmse  # Bayesian optimization minimizes
    
    def optimize(self) -> Dict[str, float]:
        """Run Bayesian optimization"""
        logger.info(f"Starting Bayesian optimization for {self.system_name}")
        logger.info(f"Using {self.config.n_calls} function evaluations with {self.config.n_initial_points} initial points")
        start_time = time.time()
        
        # Set up the decorated objective function
        @use_named_args(self.dimensions)
        def objective(**params):
            return self.objective_function([params[name] for name in self.dimension_names])
        
        try:
            # Run Bayesian optimization
            self.optimization_result = gp_minimize(
                func=objective,
                dimensions=self.dimensions,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
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


def main():
    """Example usage with different systems"""
    import sys
    from pathlib import Path
    
    if not HAS_SKOPT:
        print("Error: scikit-optimize is required for Bayesian optimization")
        print("Install with: pip install scikit-optimize")
        sys.exit(1)
    
    PROJECT_ROOT = Path.cwd()
    CONFIG_DIR = PROJECT_ROOT / "config"
    BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
    
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
    else:
        system_name = "H2"
    
    print(f"Running Bayesian optimization for {system_name}")
    
    config = BayesianConfig(n_calls=50, n_initial_points=10)
    bayes = GeneralParameterBayesian(system_name, str(BASE_PARAM_FILE), config=config)
    best_parameters = bayes.optimize()
    
    # Save results using base class methods
    bayes.save_best_parameters(bayes.system_config.optimized_params_file)
    bayes.save_fitness_history(bayes.system_config.fitness_history_file)
    
    # Save Bayesian-specific result
    result_file = bayes.system_config.optimized_params_file.replace('.toml', '_bayes_result.pkl')
    bayes.save_optimization_result(result_file)
    
    if best_parameters:
        print(f"\nBest Parameters for {system_name}:")
        for param_name, value in best_parameters.items():
            print(f"  {param_name}: {value:.6f}")


if __name__ == "__main__":
    main()
