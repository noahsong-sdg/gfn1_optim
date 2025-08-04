"""Genetic Algorithm optimizer using PyGAD for TBLite parameter optimization."""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from base_optimizer import BaseOptimizer
import pygad
import logging
from common import RANDOM_SEED

logger = logging.getLogger(__name__)

@dataclass
class PyGADConfig:
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_rate: float = 0.1
    mutation_strength: float = 0.05
    max_workers: int = 8
    convergence_threshold: float = 1e-6
    patience: int = 20
    # PyGAD specific parameters
    mutation_type: str = "random"  # "random", "gaussian", "swap", "inversion"
    crossover_type: str = "single_point"  # "single_point", "two_points", "uniform"
    selection_type: str = "tournament"  # "tournament", "roulette_wheel", "rank"


class PyGADOptimizer(BaseOptimizer):
    def __init__(self, system_name: str, base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: PyGADConfig = PyGADConfig(), train_fraction: float = 0.8,
                 spin: int = 0):
        
        self.config = config
        self.parameter_names = []
        self.parameter_bounds_list = []
        self.ga_instance = None
        
        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin)
        
        # Prepare parameter bounds for PyGAD
        self._prepare_pygad_parameters()
        
    def _prepare_pygad_parameters(self):
        """Convert parameter bounds to PyGAD format"""
        self.parameter_names = [bound.name for bound in self.parameter_bounds]
        self.parameter_bounds_list = [(bound.min_val, bound.max_val) for bound in self.parameter_bounds]
        
        logger.info(f"Prepared {len(self.parameter_names)} parameters for PyGAD optimization")
    
    def _parameters_to_dict(self, solution: np.ndarray) -> Dict[str, float]:
        """Convert PyGAD solution array to parameter dictionary"""
        return {name: value for name, value in zip(self.parameter_names, solution)}
    
    def _dict_to_parameters(self, param_dict: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to PyGAD solution array"""
        return np.array([param_dict[name] for name in self.parameter_names])
    
    def fitness_function(self, ga_instance, solution, solution_idx):
        """
        PyGAD fitness function. Converts solution array to parameter dict and evaluates.
        Note: PyGAD maximizes fitness, so we return negative RMSE.
        """
        try:
            param_dict = self._parameters_to_dict(solution)
            rmse = self.evaluate_fitness(param_dict)
            
            # Convert RMSE to fitness (PyGAD maximizes, so return negative RMSE)
            fitness = 1.0 / (1.0 + rmse)
            
            # Update best parameters if this is better
            if fitness > self.best_fitness:
                self.best_parameters = param_dict.copy()
                self.best_fitness = fitness
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0  # Return very low fitness for failed evaluations
    
    def on_generation(self, ga_instance):
        """Callback function called after each generation"""
        best_solution = ga_instance.best_solution()
        best_fitness = best_solution[1]
        
        # Record fitness history
        self.fitness_history.append({
            'generation': ga_instance.generations_completed,
            'best_fitness': best_fitness,
            'avg_fitness': np.mean(ga_instance.last_generation_fitness),
            'std_fitness': np.std(ga_instance.last_generation_fitness)
        })
        
        # Check convergence
        if len(self.fitness_history) >= 2:
            recent_improvement = abs(
                self.fitness_history[-1]['best_fitness'] - 
                self.fitness_history[-2]['best_fitness']
            )
            if recent_improvement < self.config.convergence_threshold:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0
        
        # Save checkpoint periodically
        if ga_instance.generations_completed % 10 == 0:
            self.save_checkpoint()
        
        logger.info(f"Generation {ga_instance.generations_completed}: "
                   f"Best fitness = {best_fitness:.6f}, "
                   f"Avg fitness = {np.mean(ga_instance.last_generation_fitness):.6f}")
    
    def _create_ga_instance(self):
        """Create and configure PyGAD instance"""
        
        # Create initial population with some individuals near defaults
        initial_population = []
        for i in range(self.config.population_size):
            if i == 0:
                # First individual uses default values
                default_solution = self._dict_to_parameters(
                    {bound.name: bound.default_val for bound in self.parameter_bounds}
                )
                initial_population.append(default_solution)
            else:
                # Other individuals are random but biased toward defaults
                solution = []
                for bound in self.parameter_bounds:
                    if random.random() < 0.8:  # 80% chance to stay near default
                        range_size = bound.max_val - bound.min_val
                        std = max(range_size * 0.1, 1e-6)
                        value = np.random.normal(bound.default_val, std)
                    else:
                        value = random.uniform(bound.min_val, bound.max_val)
                    solution.append(value)
                initial_population.append(solution)
        
        initial_population = np.array(initial_population)
        
        # Configure mutation
        if self.config.mutation_type == "gaussian":
            mutation_kwargs = {
                "mutation_type": "random",
                "mutation_percent_genes": self.config.mutation_rate * 100,
                "random_mutation_min_val": -self.config.mutation_strength,
                "random_mutation_max_val": self.config.mutation_strength
            }
        else:
            mutation_kwargs = {
                "mutation_type": self.config.mutation_type,
                "mutation_percent_genes": self.config.mutation_rate * 100
            }
        
        # Configure crossover
        crossover_kwargs = {
            "crossover_type": self.config.crossover_type,
            "crossover_probability": self.config.crossover_rate
        }
        
        # Create GA instance with correct PyGAD parameters
        self.ga_instance = pygad.GA(
            num_generations=self.config.generations,
            num_parents_mating=self.config.population_size // 2,
            initial_population=initial_population,
            fitness_func=self.fitness_function,
            on_generation=self.on_generation,
            gene_space=self.parameter_bounds_list,
            gene_type=float,
            random_mutation_min_val=-self.config.mutation_strength,
            random_mutation_max_val=self.config.mutation_strength,
            **mutation_kwargs,
            **crossover_kwargs,
            random_seed=RANDOM_SEED,
            parallel_processing=["thread", self.config.max_workers] if self.config.max_workers > 1 else None
        )
    
    def get_state(self) -> dict:
        state = super().get_state()
        if self.ga_instance is not None:
            state.update({
                'ga_instance': self.ga_instance,
                'parameter_names': self.parameter_names,
                'parameter_bounds_list': self.parameter_bounds_list,
                'config': self.config
            })
        return state

    def set_state(self, state: dict):
        super().set_state(state)
        self.ga_instance = state.get('ga_instance')
        self.parameter_names = state.get('parameter_names', [])
        self.parameter_bounds_list = state.get('parameter_bounds_list', [])
        self.config = state.get('config', self.config)
    
    def optimize(self) -> Dict[str, float]:
        """Run PyGAD optimization"""
        import time
        
        logger.info(f"Starting PyGAD optimization for {self.system_name}")
        logger.info(f"Population size: {self.config.population_size}")
        logger.info(f"Generations: {self.config.generations}")
        logger.info(f"Parameters to optimize: {len(self.parameter_names)}")
        
        start_time = time.time()
        
        # Create GA instance
        self._create_ga_instance()
        
        # Run optimization
        self.ga_instance.run()
        
        # Get best solution
        best_solution, best_fitness, _ = self.ga_instance.best_solution()
        best_parameters = self._parameters_to_dict(best_solution)
        
        # Set the final best parameters and fitness for reporting
        self.best_parameters = best_parameters
        self.best_fitness = self.best_rmse  # Use the tracked RMSE value
        
        optimization_time = time.time() - start_time
        
        logger.info(f"PyGAD optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best PyGAD fitness: {best_fitness:.6f}")
        logger.info(f"Best RMSE: {self.best_fitness:.6f}")
        logger.info(f"Total failed evaluations: {self.failed_evaluations}")
        
        # Save results
        self.save_best_parameters()
        self.save_fitness_history()
        
        return self.best_parameters 
