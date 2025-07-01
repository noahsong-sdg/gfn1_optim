"""
Genetic Algorithm for TBLite parameter optimization - Refactored to use BaseOptimizer
"""

import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
import pandas as pd

from base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

@dataclass
class GAConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 30  # Reduced for focused parameter set
    generations: int = 150     # Increased for better convergence
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_rate: float = 0.1
    mutation_strength: float = 0.05
    max_workers: int = 4
    convergence_threshold: float = 1e-6
    patience: int = 20


class Individual:
    """Represents a single parameter set (genome) in the GA population"""
    
    def __init__(self, parameters: Dict[str, float], fitness: float = 0.0):
        self.parameters = parameters.copy()
        self.fitness = fitness
        self.age = 0
        
    def copy(self):
        """Create a deep copy of the individual"""
        return Individual(self.parameters.copy(), self.fitness)
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.6f}, age={self.age})"


class GeneralParameterGA(BaseOptimizer):
    """Genetic Algorithm optimizer inheriting from BaseOptimizer"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: GAConfig = GAConfig(),
                 train_fraction: float = 0.8):
        """Initialize GA optimizer"""
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction)
        
        # GA-specific configuration
        self.config = config
        
        # GA-specific state
        self.population = []
        self.best_individual = None
        self.generation = 0
        
    def create_individual(self, parameters: Optional[Dict[str, float]] = None) -> Individual:
        """Create a new individual with given or random parameters"""
        if parameters is None:
            parameters = {}
            for bound in self.parameter_bounds:
                if bound.max_val <= bound.min_val:
                    parameters[bound.name] = bound.default_val
                    continue
                
                if random.random() < 0.8:  # 80% chance to stay near default
                    range_size = bound.max_val - bound.min_val
                    std = max(range_size * 0.1, 1e-6)
                    value = np.random.normal(bound.default_val, std)
                else:
                    value = random.uniform(bound.min_val, bound.max_val)
                
                value = max(bound.min_val, min(bound.max_val, value))
                parameters[bound.name] = value
        
        return Individual(parameters)
    
    def evaluate_individual_fitness(self, individual: Individual) -> float:
        """Evaluate fitness of an individual (wrapper around base class method)"""
        rmse = self.evaluate_fitness(individual.parameters)
        return 1.0 / (1.0 + rmse)  # Convert RMSE to fitness (higher is better)
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1_params = {}
        child2_params = {}
        
        for param_name in parent1.parameters:
            if random.random() < 0.5:
                child1_params[param_name] = parent1.parameters[param_name]
                child2_params[param_name] = parent2.parameters[param_name]
            else:
                child1_params[param_name] = parent2.parameters[param_name]
                child2_params[param_name] = parent1.parameters[param_name]
        
        return Individual(child1_params), Individual(child2_params)
    
    def mutate(self, individual: Individual):
        """Gaussian mutation"""
        for param_name in individual.parameters:
            if random.random() < self.config.mutation_rate:
                bound = next(b for b in self.parameter_bounds if b.name == param_name)
                mutation_std = (bound.max_val - bound.min_val) * self.config.mutation_strength
                mutation = np.random.normal(0, mutation_std)
                individual.parameters[param_name] += mutation
                
                # Apply bounds
                individual.parameters[param_name] = max(
                    bound.min_val, min(bound.max_val, individual.parameters[param_name])
                )
    
    def evolve_generation(self):
        """Evolve one generation"""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best individual
        current_best = self.population[0]
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
            self.convergence_counter = 0
        else:
            self.convergence_counter += 1
        
        # Record fitness history
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': current_best.fitness,
            'avg_fitness': np.mean([ind.fitness for ind in self.population]),
            'std_fitness': np.std([ind.fitness for ind in self.population])
        })
        
        # Elitism: keep best individuals
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        new_population = self.population[:elite_count].copy()
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)
            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
        
        # Age the population
        for individual in self.population:
            individual.age += 1
    
    def optimize(self) -> Dict[str, float]:
        """Run the genetic algorithm optimization"""
        logger.info(f"Starting genetic algorithm optimization for {self.system_name}")
        start_time = time.time()
        
        # Initialize population
        default_params = {bound.name: bound.default_val for bound in self.parameter_bounds}
        self.population = [self.create_individual(default_params)]
        for _ in range(self.config.population_size - 1):
            self.population.append(self.create_individual())
        
        for generation in range(self.config.generations):
            self.generation = generation
            logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate fitness
            for individual in self.population:
                individual.fitness = self.evaluate_individual_fitness(individual)
            
            best_fitness = max(ind.fitness for ind in self.population)
            logger.info(f"  Best fitness: {best_fitness:.6f}")
            
            # Early stopping if all fitness values are 0
            if best_fitness == 0.0:
                logger.error("All individuals have zero fitness - stopping optimization")
                break
            
            # Check convergence
            if len(self.fitness_history) >= 2:
                recent_improvement = abs(
                    self.fitness_history[-1]['best_fitness'] - 
                    self.fitness_history[-2]['best_fitness']
                )
                if recent_improvement < self.config.convergence_threshold:
                    if self.convergence_counter >= self.config.patience:
                        logger.info(f"Converged at generation {generation + 1}")
                        break
            
            # Evolve
            self.evolve_generation()
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        
        # Set best parameters for base class
        if self.best_individual is not None:
            self.best_parameters = self.best_individual.parameters.copy()
            # Convert GA fitness back to RMSE for base class consistency
            ga_fitness = self.best_individual.fitness
            self.best_fitness = (1.0 / ga_fitness) - 1.0 if ga_fitness > 0 else float('inf')
        
        return self.best_parameters


def main():
    """Example usage with different systems"""
    import sys
    from pathlib import Path
    
    PROJECT_ROOT = Path.cwd()
    CONFIG_DIR = PROJECT_ROOT / "config"
    BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
    
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
    else:
        system_name = "H2"
    
    print(f"Running GA optimization for {system_name}")
    
    config = GAConfig(population_size=30, generations=50)
    ga = GeneralParameterGA(system_name, str(BASE_PARAM_FILE), config=config)
    best_parameters = ga.optimize()
    
    # Save results using base class methods
    ga.save_best_parameters(ga.system_config.optimized_params_file)
    ga.save_fitness_history(ga.system_config.fitness_history_file)
    
    if best_parameters:
        print(f"\nBest Parameters for {system_name}:")
        for param_name, value in best_parameters.items():
            print(f"  {param_name}: {value:.6f}")


if __name__ == "__main__":
    main()
