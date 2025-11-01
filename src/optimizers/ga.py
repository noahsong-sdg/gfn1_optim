"""Genetic Algorithm optimizer for TBLite parameter optimization using DEAP."""

import numpy as np
import random
import copy
import multiprocessing
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from base_optimizer import BaseOptimizer
from common import setup_logging

from deap import base
from deap import creator
from deap import tools

logger = setup_logging(module_name="ga")

# Initialize DEAP types at module level for type hints
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax, age=0, param_dict=None)

# Global variable to hold optimizer instance for multiprocessing
# This is set/unset during optimization to enable picklable evaluation
_global_optimizer = None

def _evaluate_wrapper(individual):
    """Picklable wrapper for evaluation function used with multiprocessing."""
    assert _global_optimizer is not None, "Global optimizer not set"
    return _global_optimizer._evaluate_individual_deap(individual)

@dataclass
class GAConfig:
    population_size: int = 10  
    generations: int = 20     
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_rate: float = 0.1
    mutation_strength: float = 0.05
    max_workers: int = 12
    convergence_threshold: float = 1e-6
    patience: int = 20


class GeneralParameterGA(BaseOptimizer):
    def __init__(self, system_name: str, base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: GAConfig = GAConfig(), train_fraction: float = 0.8,
                 spin: int = 0):
        
        self.config = config
        self.generation = 0
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin)
        
        # Setup DEAP toolbox
        self._setup_toolbox()
        
        # Initialize population (empty, will be created in optimize if needed)
        self.population = []
        self.hof = tools.HallOfFame(1)  # Track best individual
        self.best_fitness_value = -float('inf')  # Track fitness (higher is better)
        
        # Initialize multiprocessing pool for parallel evaluation
        self._setup_pool()
        
    def _setup_toolbox(self):
        """Setup DEAP toolbox with registered operators."""
        self.toolbox = base.Toolbox()
        
        # Get parameter names and bounds for ordering
        self.param_names = [bound.name for bound in self.parameter_bounds]
        self.param_bounds = {bound.name: (bound.min_val, bound.max_val) for bound in self.parameter_bounds}
        self.param_defaults = {bound.name: bound.default_val for bound in self.parameter_bounds}
        
        # Register individual creation
        self.toolbox.register("individual", self._create_individual_deap)
        # Register population creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Register evaluation - use wrapper for multiprocessing compatibility
        if self.config.max_workers > 1:
            # For multiprocessing, we'll use the global wrapper
            self.toolbox.register("evaluate", _evaluate_wrapper)
        else:
            # For serial execution, we can use the instance method directly
            self.toolbox.register("evaluate", self._evaluate_individual_deap)
        # Register crossover (uniform crossover for dict-based params)
        self.toolbox.register("mate", self._crossover_deap)
        # Register mutation
        self.toolbox.register("mutate", self._mutate_deap)
        # Register selection
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        # Register clone
        self.toolbox.register("clone", copy.deepcopy)
    
    def _setup_pool(self):
        """Setup multiprocessing pool for parallel evaluation."""
        # Cleanup existing pool if any
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            except Exception:
                pass
            finally:
                self.pool = None
        
        # Create new pool if parallel execution is requested
        if self.config.max_workers > 1:
            try:
                self.pool = multiprocessing.Pool(processes=self.config.max_workers)
                self.toolbox.register("map", self.pool.map)
                # Update evaluate to use wrapper for multiprocessing
                self.toolbox.register("evaluate", _evaluate_wrapper)
            except Exception as e:
                # Fallback to serial execution if multiprocessing fails
                logger.warning(f"Failed to create multiprocessing pool: {e}. Falling back to serial execution.")
                self.pool = None
                self.toolbox.register("map", map)  # Serial execution
                self.toolbox.register("evaluate", self._evaluate_individual_deap)  # Back to instance method
        else:
            self.toolbox.register("map", map)  # Serial execution
            self.toolbox.register("evaluate", self._evaluate_individual_deap)  # Instance method
        
    def _list_to_param_dict(self, param_list: List[float]) -> Dict[str, float]:
        """Convert list of parameter values to dictionary."""
        return {name: val for name, val in zip(self.param_names, param_list)}
    
    def _param_dict_to_list(self, param_dict: Dict[str, float]) -> List[float]:
        """Convert parameter dictionary to ordered list."""
        return [param_dict[name] for name in self.param_names]
    
    def _create_individual_deap(self, parameters: Optional[Dict[str, float]] = None) -> creator.Individual:
        """Create a DEAP individual from parameters or generate random."""
        if parameters is None:
            parameters = {}
            for bound in self.parameter_bounds:
                if random.random() < 0.8:  # 80% chance to stay near default
                    range_size = bound.max_val - bound.min_val
                    std = max(range_size * 0.1, 1e-6)
                    value = np.random.normal(bound.default_val, std)
                else:
                    value = random.uniform(bound.min_val, bound.max_val)
                parameters[bound.name] = value
            
            parameters = self.apply_bounds(parameters)
        
        param_list = self._param_dict_to_list(parameters)
        individual = creator.Individual(param_list)
        individual.age = 0
        individual.param_dict = parameters.copy()
        return individual
    
    def _evaluate_individual_deap(self, individual: creator.Individual) -> Tuple[float]:
        """Evaluate fitness of a DEAP individual."""
        # Convert list to parameter dict if needed
        if individual.param_dict is None:
            individual.param_dict = self._list_to_param_dict(individual)
        
        # Ensure bounds are applied
        individual.param_dict = self.apply_bounds(individual.param_dict)
        
        # Update the list representation
        param_list = self._param_dict_to_list(individual.param_dict)
        individual[:] = param_list
        
        # Evaluate fitness
        fitness_val = self.evaluate_fitness(individual.param_dict)
        return (fitness_val,)  # DEAP expects tuple
    
    def _crossover_deap(self, ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
        """Uniform crossover for DEAP individuals."""
        if random.random() > self.config.crossover_rate:
            return ind1, ind2
        
        # Perform uniform crossover (similar to original implementation)
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        
        # Update param_dict for both individuals
        ind1.param_dict = self._list_to_param_dict(ind1)
        ind2.param_dict = self._list_to_param_dict(ind2)
        
        # Invalidate fitness
        del ind1.fitness.values
        del ind2.fitness.values
        
        return ind1, ind2
    
    def _mutate_deap(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """Gaussian mutation for DEAP individual."""
        # Convert to param dict if needed
        if individual.param_dict is None:
            individual.param_dict = self._list_to_param_dict(individual)
        
        # Apply mutation
        for i, param_name in enumerate(self.param_names):
            if random.random() < self.config.mutation_rate:
                bound = next(b for b in self.parameter_bounds if b.name == param_name)
                mutation_std = (bound.max_val - bound.min_val) * self.config.mutation_strength
                mutation = np.random.normal(0, mutation_std)
                individual[i] += mutation
        
        # Apply bounds using centralized system
        individual.param_dict = self._list_to_param_dict(individual)
        individual.param_dict = self.apply_bounds(individual.param_dict)
        
        # Update list representation
        param_list = self._param_dict_to_list(individual.param_dict)
        individual[:] = param_list
        
        # Invalidate fitness
        del individual.fitness.values
        
        return (individual,)
    
    def _evolve_generation_deap(self):
        """Evolve one generation using DEAP operators."""
        # Evaluate individuals
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update hall of fame
        self.hof.update(self.population)
        
        # Track best individual
        current_best = self.hof[0]
        current_fitness = current_best.fitness.values[0]
        if self.best_parameters is None or current_fitness > self.best_fitness_value:
            self.best_parameters = current_best.param_dict.copy()
            self.best_fitness_value = current_fitness
            self.convergence_counter = 0
        else:
            self.convergence_counter += 1
        
        # Record fitness history
        fitnesses_list = [ind.fitness.values[0] for ind in self.population]
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': max(fitnesses_list),
            'avg_fitness': np.mean(fitnesses_list),
            'std_fitness': np.std(fitnesses_list)
        })
        
        # Elitism: keep best individuals
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        elites = sorted(self.population, key=lambda x: x.fitness.values[0], reverse=True)[:elite_count]
        
        # Select parents for offspring
        offspring = self.toolbox.select(self.population, len(self.population) - elite_count)
        offspring = [self.toolbox.clone(ind) for ind in offspring]
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.config.crossover_rate:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            self.toolbox.mutate(mutant)
        
        # Replace population with elites + offspring
        self.population = elites + offspring
        self.population = self.population[:self.config.population_size]
        
        # Age the population
        for individual in self.population:
            individual.age += 1
    
    def get_state(self) -> dict:
        """Get state for checkpointing."""
        state = super().get_state()
        
        # Serialize population as parameter dicts (DEAP individuals are complex)
        population_dicts = []
        for ind in self.population:
            if ind.param_dict is not None:
                population_dicts.append(ind.param_dict.copy())
            else:
                population_dicts.append(self._list_to_param_dict(list(ind)))
        
        # Serialize hall of fame
        hof_dict = None
        if len(self.hof) > 0:
            if self.hof[0].param_dict is not None:
                hof_dict = self.hof[0].param_dict.copy()
            else:
                hof_dict = self._list_to_param_dict(list(self.hof[0]))
        
        state.update({
            'population_dicts': population_dicts,
            'hof_dict': hof_dict,
            'generation': self.generation,
            'config': self.config,
            'best_fitness_value': self.best_fitness_value
        })
        return state

    def set_state(self, state: dict):
        """Set state from checkpoint."""
        super().set_state(state)
        
        # Reconstruct population from parameter dicts
        population_dicts = state.get('population_dicts', [])
        self.population = []
        for param_dict in population_dicts:
            individual = self._create_individual_deap(param_dict)
            self.population.append(individual)
        
        # Reconstruct hall of fame
        hof_dict = state.get('hof_dict')
        if hof_dict is not None:
            hof_ind = self._create_individual_deap(hof_dict)
            # Evaluate to set fitness
            hof_ind.fitness.values = self.toolbox.evaluate(hof_ind)
            self.hof = tools.HallOfFame(1)
            self.hof.insert(hof_ind)
        
        self.generation = state.get('generation', 0)
        self.config = state.get('config', self.config)
        self.best_fitness_value = state.get('best_fitness_value', -float('inf'))
        
        # Recreate multiprocessing pool after checkpoint load (pool can't be pickled)
        self._setup_pool()
    
    def optimize(self) -> Dict[str, float]:
        """Run the genetic algorithm optimization."""
        # Ensure pool is set up (in case optimize is called directly without checkpoint)
        if not hasattr(self, 'pool') or (self.config.max_workers > 1 and self.pool is None):
            self._setup_pool()
        
        # Set global optimizer for multiprocessing (needed for picklable evaluation)
        global _global_optimizer
        old_optimizer = _global_optimizer
        _global_optimizer = self
        
        try:
            # Initialize population if empty
        if not self.population:
            default_params = {bound.name: bound.default_val for bound in self.parameter_bounds}
                self.population = [self._create_individual_deap(default_params)]
            for _ in range(self.config.population_size - 1):
                    self.population.append(self._create_individual_deap())
        
            # Main evolution loop
        for generation in range(self.generation, self.config.generations):
            self.generation = generation
            
                # Evaluate all individuals
                invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Check for perfect fitness
                best_fitness = max(ind.fitness.values[0] for ind in self.population)
            if best_fitness == 0.0:
                break
            
                # Check convergence
            if len(self.fitness_history) >= 2:
                recent_improvement = abs(
                    self.fitness_history[-1]['best_fitness'] - 
                    self.fitness_history[-2]['best_fitness']
                )
                if recent_improvement < self.config.convergence_threshold:
                    if self.convergence_counter >= self.config.patience:
                        break
            
                # Evolve generation
                self._evolve_generation_deap()
            self.save_checkpoint()
        
            # Finalize best parameters
            if self.hof and len(self.hof) > 0:
                best_ind = self.hof[0]
                if best_ind.param_dict is not None:
                    self.best_parameters = best_ind.param_dict.copy()
                else:
                    self.best_parameters = self._list_to_param_dict(list(best_ind))
                ga_fitness = best_ind.fitness.values[0]
                self.best_fitness_value = ga_fitness
            elif self.best_parameters is not None:
                # Use tracked best parameters
                pass
            else:
                # Fallback: use best from current population
                best_ind = max(self.population, key=lambda x: x.fitness.values[0])
                if best_ind.param_dict is not None:
                    self.best_parameters = best_ind.param_dict.copy()
                else:
                    self.best_parameters = self._list_to_param_dict(list(best_ind))
                ga_fitness = best_ind.fitness.values[0]
                self.best_fitness_value = ga_fitness
            
            # Convert best_fitness_value from fitness to RMSE for consistency with base class
            if self.best_fitness_value > 0:
                self.best_fitness = (1.0 / self.best_fitness_value) - 1.0
            else:
                self.best_fitness = float('inf')
            
            # Cleanup multiprocessing pool
            if hasattr(self, 'pool') and self.pool is not None:
                try:
                    self.pool.close()
                    self.pool.join()
                except Exception:
                    pass  # Ignore errors during cleanup
                finally:
                    self.pool = None
        
        return self.best_parameters
        finally:
            # Restore global optimizer
            _global_optimizer = old_optimizer
    
    def __del__(self):
        """Cleanup pool on object deletion."""
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception:
                pass
