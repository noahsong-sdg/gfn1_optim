"""Particle Swarm Optimization for TBLite parameter optimization."""

import numpy as np
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from base_optimizer import BaseOptimizer
from parameter_bounds import ParameterBounds

@dataclass
class PSOConfig:
    swarm_size: int = 40
    max_iterations: int = 100
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive coefficient
    c2: float = 1.5  # Social coefficient
    w_min: float = 0.1  # Minimum inertia weight
    w_max: float = 0.9  # Maximum inertia weight
    use_adaptive_inertia: bool = True
    convergence_threshold: float = 1e-6
    patience: int = 20


class Particle:
    def __init__(self, position: Dict[str, float], parameter_bounds):
        self.position = position.copy()
        self.velocity = {name: 0.0 for name in position.keys()}
        self.best_position = position.copy()
        self.fitness = 0.0
        self.best_fitness = 0.0
        self.parameter_bounds = parameter_bounds
        
    def update_velocity(self, global_best_position: Dict[str, float], w: float, c1: float, c2: float):
        """Update particle velocity"""
        for param_name in self.position:
            # Get parameter bounds
            bound = next(b for b in self.parameter_bounds if b.name == param_name)
            max_velocity = (bound.max_val - bound.min_val) * 0.2  # Limit velocity to 20% of parameter range
            
            # Calculate velocity components
            cognitive = c1 * np.random.random() * (self.best_position[param_name] - self.position[param_name])
            social = c2 * np.random.random() * (global_best_position[param_name] - self.position[param_name])
            
            # Update velocity with inertia
            self.velocity[param_name] = (w * self.velocity[param_name] + cognitive + social)
            
            # Clamp velocity
            self.velocity[param_name] = max(-max_velocity, min(max_velocity, self.velocity[param_name]))
    
    def update_position(self):
        """Update particle position"""
        for param_name in self.position:
            # Update position
            self.position[param_name] += self.velocity[param_name]
    
    def update_best(self):
        """Update personal best if current fitness is better"""
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()


class GeneralParameterPSO(BaseOptimizer):
    """Particle Swarm Optimization optimizer inheriting from BaseOptimizer"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: PSOConfig = PSOConfig(),
                 train_fraction: float = 0.8):
        """Initialize PSO optimizer"""
        
        # PSO-specific configuration (set before super().__init__ to avoid set_state issues)
        self.config = config
        
        # PSO-specific state
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = 0.0
        self.iteration = 0
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction)
        
    def initialize_particle(self, parameters: Optional[Dict[str, float]] = None) -> Particle:
        """Initialize a single particle"""
        if parameters is None:
            parameters = {}
            for bound in self.parameter_bounds:
                if np.random.random() < 0.8:  # 80% chance to stay near default
                    range_size = bound.max_val - bound.min_val
                    std = max(range_size * 0.1, 1e-6)
                    value = np.random.normal(bound.default_val, std)
                else:
                    value = np.random.uniform(bound.min_val, bound.max_val)
                
                parameters[bound.name] = value
            
            # Apply bounds using centralized system
            parameters = self.apply_bounds(parameters)
        
        return Particle(parameters, self.parameter_bounds)
    
    def evaluate_particle_fitness(self, particle: Particle) -> float:
        """Evaluate fitness of a particle (wrapper around base class method)"""
        rmse = self.evaluate_fitness(particle.position)
        return 1.0 / (1.0 + rmse)  # Convert RMSE to fitness (higher is better)
    
    def initialize_swarm(self):
        """Initialize the swarm"""
        logger.info(f"Initializing swarm of size {self.config.swarm_size}")
        
        # Include one particle with default parameters
        default_params = {bound.name: bound.default_val for bound in self.parameter_bounds}
        self.swarm = [self.initialize_particle(default_params)]
        
        # Add random particles
        for _ in range(self.config.swarm_size - 1):
            self.swarm.append(self.initialize_particle())
    
    def update_global_best(self):
        """Update global best position"""
        for particle in self.swarm:
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
                self.convergence_counter = 0
                logger.info(f"  NEW GLOBAL BEST at iteration {self.iteration + 1}: fitness = {self.global_best_fitness:.6f}")
            else:
                if hasattr(self, 'convergence_counter'):
                    self.convergence_counter += 1
                else:
                    self.convergence_counter = 0
    
    def get_adaptive_inertia(self) -> float:
        """Calculate adaptive inertia weight"""
        if not self.config.use_adaptive_inertia:
            return self.config.w
        
        # Linear decrease from w_max to w_min
        progress = self.iteration / self.config.max_iterations
        return self.config.w_max - (self.config.w_max - self.config.w_min) * progress
    
    def optimize(self) -> Dict[str, float]:
        """Run PSO optimization"""
        logger.info(f"Starting PSO optimization for {self.system_name}")
        start_time = time.time()
        
        # Initialize swarm
        self.initialize_swarm()
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Evaluate fitness for all particles
            for particle in self.swarm:
                particle.fitness = self.evaluate_particle_fitness(particle)
                particle.update_best()
            
            # Update global best
            self.update_global_best()
            
            best_fitness = max(p.fitness for p in self.swarm)
            avg_fitness = np.mean([p.fitness for p in self.swarm])
            logger.info(f"  Best fitness: {best_fitness:.6f}, Avg: {avg_fitness:.6f}")
            
            # Record fitness history
            self.fitness_history.append({
                'generation': iteration,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std([p.fitness for p in self.swarm])
            })
            
            # Early stopping if all fitness values are 0
            if best_fitness == 0.0:
                logger.error("All particles have zero fitness - stopping optimization")
                break
            
            # Check convergence
            if len(self.fitness_history) >= 2:
                recent_improvement = abs(
                    self.fitness_history[-1]['best_fitness'] - 
                    self.fitness_history[-2]['best_fitness']
                )
                if recent_improvement < self.config.convergence_threshold:
                    if self.convergence_counter >= self.config.patience:
                        logger.info(f"Converged at iteration {iteration + 1}")
                        break
            
            # Update particles
            if self.global_best_position is not None:
                inertia = self.get_adaptive_inertia()
                
                for particle in self.swarm:
                    particle.update_velocity(
                        self.global_best_position, 
                        inertia, 
                        self.config.c1, 
                        self.config.c2
                    )
                    particle.update_position()
                    # Apply bounds using centralized system
                    particle.position = self.apply_bounds(particle.position)
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        
        # Set best parameters for base class
        if self.global_best_position is not None:
            self.best_parameters = self.global_best_position.copy()
            self.best_fitness = self.global_best_fitness
        
        return self.best_parameters

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            'swarm': self.swarm,
            'global_best_position': self.global_best_position,
            'global_best_fitness': self.global_best_fitness,
            'iteration': self.iteration,
            'config': self.config
        })
        return state

    def set_state(self, state: dict):
        super().set_state(state)
        self.swarm = state.get('swarm', [])
        self.global_best_position = state.get('global_best_position')
        self.global_best_fitness = state.get('global_best_fitness', 0.0)
        self.iteration = state.get('iteration', 0)
        self.config = state.get('config', self.config)


 
