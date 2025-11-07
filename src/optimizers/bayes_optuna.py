"""
Optuna-based Bayesian Optimization for TBLite parameter optimization.

This module mirrors the public API of `bayes_h.py` but uses Optuna
for Bayesian optimization with parallelization support.
"""

import logging
import time
import threading
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@dataclass
class BayesianConfig:
    """Configuration for Optuna-based Bayesian optimization.

    Fields largely mirror the `bayes_h.BayesianConfig` for API compatibility.
    Some are unused in Optuna but retained to avoid breaking callers.
    """
    n_calls: int = 500
    n_initial_points: int = 20  # mapped to TPE's n_startup_trials when not resuming
    acq_func: str = "EI"  # unused (kept for compatibility)
    acq_optimizer: str = "auto"  # unused (kept for compatibility)
    xi: float = 0.1  # unused (kept for compatibility)
    kappa: float = 1.96  # unused (kept for compatibility)
    n_restarts_optimizer: int = 5  # unused (kept for compatibility)
    noise: float = 1e-10  # unused (kept for compatibility)
    random_state: Optional[int] = None
    # Optuna-specific
    n_jobs: int = 1  # parallel trials
    sampler: str = "TPE"  # only TPE is configured by default


class GeneralParameterBayesian(BaseOptimizer):
    """Optuna-based Bayesian Optimization optimizer inheriting from BaseOptimizer."""

    def __init__(self,
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: BayesianConfig = BayesianConfig(),
                 train_fraction: float = 0.8,
                 spin: int = 0,
                 **kwargs):
        # Optuna-specific configuration/state first (avoid set_state issues)
        self.config = config
        self.optimization_result = None  # Will store convergence-friendly dict
        self.call_count = 0
        self._lock = threading.Lock()  # Ensure thread-safe updates in parallel runs

        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin, **kwargs)

        # Prepare search space meta
        self.dimensions = self._create_search_space()
        if hasattr(self, '_saved_dimension_names') and self._saved_dimension_names is not None:
            self.dimension_names = self._saved_dimension_names
            logger.debug("Using saved dimension names from checkpoint")
        else:
            self.dimension_names = [bound.name for bound in self.parameter_bounds]
            logger.debug("Creating new dimension names")

        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for GeneralParameterBayesian in bayes_h_optuna.py")

        # Create study (in-memory by default). If persistence is desired, callers
        # can extend config to include a storage URL and study name.
        sampler = self._create_sampler()
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

    def _create_sampler(self):
        # Only TPE is used by default; maps n_initial_points to n_startup_trials when fresh
        n_startup = max(0, int(self.config.n_initial_points))
        seed = self.config.random_state
        return TPESampler(seed=seed, n_startup_trials=n_startup)

    def _create_search_space(self) -> List[Tuple[str, float, float]]:
        """Create search space as list of (name, low, high) tuples."""
        dimensions = []
        logger.debug(f"Creating search space for {len(self.parameter_bounds)} parameters")

        # Validate bounds centrally
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
            dimensions.append((bound.name, float(bound.min_val), float(bound.max_val)))

        logger.info(f"Created search space with {len(dimensions)} dimensions")
        return dimensions

    def objective_function(self, x: List[float]) -> float:
        """Objective function for Optuna (minimizes RMSE)."""
        # Convert vector to dict aligned with dimension_names
        parameters = {name: value for name, value in zip(self.dimension_names, x)}

        # Evaluate fitness using base class (higher is better)
        fitness = self.evaluate_fitness(parameters)
        rmse = (1.0 / fitness) - 1.0 if fitness > 0 else float('inf')

        # Penalize failed evaluations with large finite value
        if not np.isfinite(rmse):
            rmse = 1000.0
            logger.warning("Evaluation failed, using penalty value 1000.0")

        with self._lock:
            self.call_count += 1
            # Record fitness history as RMSE (lower better)
            self.fitness_history.append({
                'generation': self.call_count - 1,
                'best_fitness': rmse,
                'avg_fitness': rmse,
                'std_fitness': 0.0
            })

            if self.call_count % 10 == 0 or rmse < self.best_fitness:
                logger.info(f"Call {self.call_count}/{self.config.n_calls}: RMSE = {rmse:.6f}")

            # Update best if improved
            if rmse < self.best_fitness:
                self.best_fitness = rmse
                self.best_parameters = parameters.copy()
                logger.info(f"  NEW BEST at call {self.call_count}: RMSE = {rmse:.6f}")

            # Save checkpoint periodically
            if self.call_count % 10 == 0:
                self.save_checkpoint()

        return rmse

    def optimize(self) -> Dict[str, float]:
        """Run Optuna optimization with optional parallelization via n_jobs."""
        logger.info(f"Starting Optuna Bayesian optimization for {self.system_name}")
        logger.info(f"Using {self.config.n_calls} total evaluations with {self.config.n_initial_points} initial points")
        start_time = time.time()

        # Attempt to load checkpoint
        resumed = False
        try:
            self.load_checkpoint()
            if self.call_count > 0:
                logger.info(f"Resuming from checkpoint at call {self.call_count}")
                resumed = True
            else:
                logger.info("Checkpoint loaded but no progress found, starting fresh")
                resumed = False
        except Exception:
            logger.info("No valid checkpoint found, starting fresh optimization")
            resumed = False

        # Update sampler startup trials when resuming (no warmup trials)
        if resumed:
            # Replace sampler to avoid random startup again
            seed = self.config.random_state
            self.study.sampler = TPESampler(seed=seed, n_startup_trials=0)

        remaining_calls = max(0, self.config.n_calls - self.call_count)
        if remaining_calls <= 0:
            logger.info("Optimization already completed based on checkpoint")
            return self.best_parameters or {}

        logger.info(f"Running {remaining_calls} remaining function evaluations")

        # If resuming and we have a best point, enqueue it so Optuna evaluates around it
        if resumed and self.best_parameters:
            try:
                self.study.enqueue_trial(self.best_parameters)
            except Exception as e:
                logger.warning(f"Failed to enqueue best_parameters for warm start: {e}")

        # Define Objective for Optuna Trial
        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, low, high in self.dimensions:
                params[name] = trial.suggest_float(name, low, high)
            x = [params[name] for name in self.dimension_names]
            return self.objective_function(x)

        try:
            self.study.optimize(
                objective,
                n_trials=remaining_calls,
                n_jobs=max(1, int(self.config.n_jobs)),
                gc_after_trial=True,
                show_progress_bar=False
            )

            # Extract best parameters
            best_rmse = float(self.study.best_value)
            best_params = dict(self.study.best_params)

            with self._lock:
                self.best_parameters = best_params
                self.best_fitness = best_rmse

            # Build a convergence-friendly snapshot
            complete_trials = [t for t in self.study.trials if t.value is not None and t.state.name == 'COMPLETE']
            func_vals = [float(t.value) for t in complete_trials]
            x_iters = [
                [float(t.params[name]) for name in self.dimension_names]
                for t in complete_trials
            ]
            self.optimization_result = {
                'func_vals': func_vals,
                'x_iters': x_iters,
                'best_params': best_params,
                'best_value': best_rmse
            }

            total_time = time.time() - start_time
            logger.info(f"Optimization completed in {total_time:.2f}s")
            logger.info(f"Best RMSE: {best_rmse:.6f}")

            # Save final checkpoint
            self.save_checkpoint()

            return self.best_parameters

        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
            raise

    def save_optimization_result(self, filename: str):
        """Save a compact snapshot of the optimization for later analysis."""
        if self.optimization_result is None:
            # Build from study if present
            if hasattr(self, 'study') and self.study is not None:
                complete_trials = [t for t in self.study.trials if t.value is not None and t.state.name == 'COMPLETE']
                func_vals = [float(t.value) for t in complete_trials]
                x_iters = [
                    [float(t.params[name]) for name in self.dimension_names]
                    for t in complete_trials
                ]
                best_params = dict(self.study.best_params) if len(complete_trials) > 0 else {}
                best_value = float(self.study.best_value) if len(complete_trials) > 0 else float('inf')
                snapshot = {
                    'func_vals': func_vals,
                    'x_iters': x_iters,
                    'best_params': best_params,
                    'best_value': best_value
                }
            else:
                raise ValueError("No optimization has been run")
        else:
            snapshot = self.optimization_result

        with open(filename, 'wb') as f:
            pickle.dump(snapshot, f)
        logger.info(f"Optimization result saved to {filename}")

    def load_optimization_result(self, filename: str):
        """Load a previously saved optimization snapshot."""
        with open(filename, 'rb') as f:
            self.optimization_result = pickle.load(f)
        logger.info(f"Optimization result loaded from {filename}")

    def get_convergence_data(self) -> Dict[str, List[float]]:
        """Get convergence data resembling skopt output format."""
        func_vals: List[float]
        x_iters: List[List[float]]

        if self.optimization_result is not None:
            func_vals = list(self.optimization_result.get('func_vals', []))
            x_iters = [list(x) for x in self.optimization_result.get('x_iters', [])]
        elif hasattr(self, 'study') and self.study is not None:
            complete_trials = [t for t in self.study.trials if t.value is not None and t.state.name == 'COMPLETE']
            func_vals = [float(t.value) for t in complete_trials]
            x_iters = [
                [float(t.params[name]) for name in self.dimension_names]
                for t in complete_trials
            ]
        else:
            raise ValueError("No optimization has been run")

        running_min: List[float] = []
        current_min = float('inf')
        for val in func_vals:
            current_min = min(current_min, val)
            running_min.append(current_min)

        return {
            'func_vals': func_vals,
            'running_min': running_min,
            'x_iters': x_iters
        }

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            'call_count': self.call_count,
            'config': self.config,
            'dimension_names': getattr(self, 'dimension_names', None)
        })

        # Exclude study/optimization_result objects from checkpoint to avoid pickling issues
        if 'optimization_result' in state:
            del state['optimization_result']
        return state

    def set_state(self, state: dict):
        super().set_state(state)
        self.call_count = state.get('call_count', 0)
        self.config = state.get('config', self.config)

        if hasattr(self, 'dimension_names'):
            self.dimension_names = state.get('dimension_names', self.dimension_names)
        else:
            self._saved_dimension_names = state.get('dimension_names', None)

        # Reset transient objects
        self.optimization_result = None
        # Re-create study with updated sampler based on config
        if HAS_OPTUNA:
            self.study = optuna.create_study(direction="minimize", sampler=self._create_sampler())


