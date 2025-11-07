"""Particle Swarm Optimization for TBLite parameter optimization using DEAP."""

import numpy as np
import random
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from base_optimizer import BaseOptimizer
from utils.parameter_bounds import ParameterBounds
from common import setup_logging
import os

from deap import base
from deap import creator
from deap import tools

logger = setup_logging(module_name="pso")

# Initialize DEAP types at module level for type hints
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Particle"):
    creator.create("Particle", list, fitness=creator.FitnessMax, 
                  speed=list, smin=None, smax=None, best=None, param_dict=None)

@dataclass
class PSOConfig:
    swarm_size: int = 10
    max_iterations: int = 20
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive coefficient
    c2: float = 1.5  # Social coefficient
    w_min: float = 0.1  # Minimum inertia weight
    w_max: float = 0.9  # Maximum inertia weight
    use_adaptive_inertia: bool = True
    convergence_threshold: float = 1e-8  # Much stricter convergence threshold
    patience: int = 20  # Much higher patience
    min_iterations: int = 20  # Minimum iterations before allowing convergence


class GeneralParameterPSO(BaseOptimizer):
    """Particle Swarm Optimization optimizer using DEAP, inheriting from BaseOptimizer"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: PSOConfig = PSOConfig(),
                 train_fraction: float = 0.8,
                 spin: int = 0,
                 **kwargs):
        """Initialize PSO optimizer"""
        
        # PSO-specific configuration (set before super().__init__ to avoid set_state issues)
        self.config = config
        self.iteration = 0
        
        # Initialize base optimizer
        super().__init__(system_name, base_param_file, reference_data, train_fraction, spin, **kwargs)
        
        # Setup DEAP toolbox
        self._setup_toolbox()
        
        # Initialize swarm (empty, will be created in optimize if needed)
        self.swarm = []
        self.global_best = None
        self.global_best_fitness = -float('inf')  # Track fitness (higher is better)
        
    def _setup_toolbox(self):
        """Setup DEAP toolbox with registered operators."""
        self.toolbox = base.Toolbox()
        
        # Get parameter names and bounds for ordering
        self.param_names = [bound.name for bound in self.parameter_bounds]
        self.param_bounds = {bound.name: (bound.min_val, bound.max_val) for bound in self.parameter_bounds}
        self.param_defaults = {bound.name: bound.default_val for bound in self.parameter_bounds}
        
        # Register individual creation
        self.toolbox.register("particle", self._create_particle_deap)
        # Register swarm creation
        self.toolbox.register("swarm", tools.initRepeat, list, self.toolbox.particle)
        # Register evaluation
        self.toolbox.register("evaluate", self._evaluate_particle_deap)
        # Register clone
        self.toolbox.register("clone", copy.deepcopy)
    
    def _list_to_param_dict(self, param_list: List[float]) -> Dict[str, float]:
        """Convert list of parameter values to dictionary."""
        return {name: val for name, val in zip(self.param_names, param_list)}
    
    def _param_dict_to_list(self, param_dict: Dict[str, float]) -> List[float]:
        """Convert parameter dictionary to ordered list."""
        return [param_dict[name] for name in self.param_names]
    
    def _create_particle_deap(self, parameters: Optional[Dict[str, float]] = None) -> creator.Particle:
        """Create a DEAP particle from parameters or generate random."""
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
        
        # Create particle with position and velocity
        particle = creator.Particle(param_list)
        
        # Initialize velocity (zero or small random)
        particle.speed = [0.0] * len(param_list)
        
        # Set velocity bounds (20% of parameter range)
        particle.smin = []
        particle.smax = []
        for name in self.param_names:
            bound = next(b for b in self.parameter_bounds if b.name == name)
            max_velocity = (bound.max_val - bound.min_val) * 0.2
            particle.smin.append(-max_velocity)
            particle.smax.append(max_velocity)
        
        # Initialize best position (same as current position)
        particle.best = creator.Particle(param_list)
        particle.best[:] = param_list
        particle.best.speed = particle.speed[:]
        particle.best.smin = particle.smin[:]
        particle.best.smax = particle.smax[:]
        particle.best.param_dict = parameters.copy()
        # Initialize best fitness with worst value (will be updated when evaluated)
        particle.best.fitness.values = (-float('inf'),)
        
        # Store parameter dict for easy access
        particle.param_dict = parameters.copy()
        
        return particle
    
    def _evaluate_particle_deap(self, particle: creator.Particle) -> Tuple[float]:
        """Evaluate fitness of a DEAP particle."""
        # Convert list to parameter dict if needed
        if particle.param_dict is None:
            particle.param_dict = self._list_to_param_dict(particle)
        
        # Ensure bounds are applied
        particle.param_dict = self.apply_bounds(particle.param_dict)
        
        # Update the list representation
        param_list = self._param_dict_to_list(particle.param_dict)
        particle[:] = param_list
        
        # Evaluate fitness
        fitness_val = self.evaluate_fitness(particle.param_dict)
        return (fitness_val,)  # DEAP expects tuple
    
    def _update_particle(self, particle: creator.Particle, global_best: creator.Particle, w: float, c1: float, c2: float):
        """Update particle velocity and position using PSO equations."""
        # Update velocity
        for i in range(len(particle)):
            r1 = random.random()
            r2 = random.random()
            
            # Calculate velocity components
            cognitive = c1 * r1 * (particle.best[i] - particle[i])
            social = c2 * r2 * (global_best[i] - particle[i])
            
            # Update velocity with inertia
            particle.speed[i] = w * particle.speed[i] + cognitive + social
            
            # Clamp velocity
            particle.speed[i] = max(particle.smin[i], min(particle.smax[i], particle.speed[i]))
        
        # Update position
        for i in range(len(particle)):
            particle[i] += particle.speed[i]
        
        # Convert to param dict and apply bounds
        particle.param_dict = self._list_to_param_dict(particle)
        particle.param_dict = self.apply_bounds(particle.param_dict)
        
        # Update list representation
        param_list = self._param_dict_to_list(particle.param_dict)
        particle[:] = param_list
        
        # Update best position if current fitness is better
        if particle.fitness.valid:
            # Initialize best fitness if not set
            if not particle.best.fitness.valid:
                particle.best.fitness.values = (-float('inf'),)
            
            if particle.fitness.values[0] > particle.best.fitness.values[0]:
                particle.best[:] = particle[:]
                particle.best.param_dict = particle.param_dict.copy()
                particle.best.fitness.values = particle.fitness.values
    
    def _get_adaptive_inertia(self) -> float:
        """Calculate adaptive inertia weight"""
        if not self.config.use_adaptive_inertia:
            return self.config.w
        
        # Linear decrease from w_max to w_min
        progress = self.iteration / self.config.max_iterations
        return self.config.w_max - (self.config.w_max - self.config.w_min) * progress
    
    def optimize(self) -> Dict[str, float]:
        """Run PSO optimization using DEAP"""
        logger.info(f"Starting PSO optimization for {self.system_name}")
        
        # Try to load checkpoint if exists
        checkpoint_path = self.get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_checkpoint()
            completed = self.iteration
            remaining = max(0, self.config.max_iterations - self.iteration)
            progress_pct = (completed / self.config.max_iterations * 100) if self.config.max_iterations > 0 else 0
            logger.info(f"Resuming PSO optimization")
            logger.info(f"  Progress: {completed}/{self.config.max_iterations} iterations completed ({progress_pct:.1f}%)")
            logger.info(f"  Remaining: {remaining} iterations to complete")
            if hasattr(self, 'global_best_fitness') and self.global_best_fitness != -float('inf'):
                logger.info(f"  Current best fitness: {self.global_best_fitness:.6f}")
        else:
            logger.info("No checkpoint found, starting fresh optimization")
            # Initialize swarm
            default_params = {bound.name: bound.default_val for bound in self.parameter_bounds}
            self.swarm = [self._create_particle_deap(default_params)]
            for _ in range(self.config.swarm_size - 1):
                self.swarm.append(self._create_particle_deap())
            
            # Initialize global best
            self.global_best = None
            self.global_best_fitness = -float('inf')
        
        for iteration in range(self.iteration, self.config.max_iterations):
            self.iteration = iteration
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Evaluate fitness for all particles
            invalid_particles = [p for p in self.swarm if not p.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_particles)
            for particle, fitness in zip(invalid_particles, fitnesses):
                particle.fitness.values = fitness
            
            # Update global best
            for particle in self.swarm:
                fitness_val = particle.fitness.values[0]
                if fitness_val > self.global_best_fitness:
                    self.global_best_fitness = fitness_val
                    self.global_best = self.toolbox.clone(particle)
                    self.convergence_counter = 0
                    logger.info(f"  NEW GLOBAL BEST at iteration {iteration + 1}: fitness = {self.global_best_fitness:.6f}")
            
            # Update personal best for each particle
            for particle in self.swarm:
                if particle.fitness.valid:
                    # Initialize best fitness if not set
                    if not particle.best.fitness.valid:
                        particle.best.fitness.values = (-float('inf'),)
                    
                    if particle.fitness.values[0] > particle.best.fitness.values[0]:
                        particle.best[:] = particle[:]
                        particle.best.param_dict = particle.param_dict.copy() if particle.param_dict else self._list_to_param_dict(particle)
                        particle.best.fitness.values = particle.fitness.values
            
            best_fitness = max(p.fitness.values[0] for p in self.swarm)
            avg_fitness = np.mean([p.fitness.values[0] for p in self.swarm])
            logger.info(f"  Best fitness: {best_fitness:.6f}, Avg: {avg_fitness:.6f}")
            
            # Record fitness history
            self.fitness_history.append({
                'generation': iteration,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std([p.fitness.values[0] for p in self.swarm])
            })
            
            # Update best parameters for base class
            if self.global_best is not None:
                if self.global_best.param_dict is not None:
                    self.best_parameters = self.global_best.param_dict.copy()
                else:
                    self.best_parameters = self._list_to_param_dict(list(self.global_best))
                # Convert fitness to RMSE for consistency with base class
                if self.global_best_fitness > 0:
                    self.best_fitness = (1.0 / self.global_best_fitness) - 1.0
                else:
                    self.best_fitness = float('inf')
            
            # Save checkpoint every 10 iterations
            if iteration % 10 == 0:
                self.save_checkpoint()
                logger.debug(f"Checkpoint saved at iteration {iteration}")
            
            # Early stopping if all fitness values are 0
            if best_fitness == 0.0:
                logger.error("All particles have zero fitness - stopping optimization")
                break
            
            # Check convergence
            if len(self.fitness_history) >= 2 and iteration >= self.config.min_iterations:
                recent_improvement = abs(
                    self.fitness_history[-1]['best_fitness'] - 
                    self.fitness_history[-2]['best_fitness']
                )
                
                # More stringent convergence criteria
                if recent_improvement < self.config.convergence_threshold:
                    self.convergence_counter += 1
                    if self.convergence_counter >= self.config.patience:
                        logger.info(f"Converged at iteration {iteration + 1} after {self.convergence_counter} iterations without improvement")
                        break
                else:
                    self.convergence_counter = 0
            
            # Update particles
            if self.global_best is not None:
                inertia = self._get_adaptive_inertia()
                
                for particle in self.swarm:
                    self._update_particle(particle, self.global_best, inertia, self.config.c1, self.config.c2)
                    # Invalidate fitness after position update
                    del particle.fitness.values
        
        # Finalize best parameters
        if self.global_best is not None:
            if self.global_best.param_dict is not None:
                self.best_parameters = self.global_best.param_dict.copy()
            else:
                self.best_parameters = self._list_to_param_dict(list(self.global_best))
            # Convert fitness to RMSE for consistency with base class
            if self.global_best_fitness > 0:
                self.best_fitness = (1.0 / self.global_best_fitness) - 1.0
            else:
                self.best_fitness = float('inf')
        
        # Save final checkpoint
        self.save_checkpoint()
        logger.info("Final checkpoint saved")
        
        return self.best_parameters

    def get_state(self) -> dict:
        """Get state for checkpointing."""
        state = super().get_state()
        
        # Serialize swarm as parameter dicts (DEAP particles are complex)
        swarm_dicts = []
        for particle in self.swarm:
            if particle.param_dict is not None:
                swarm_dicts.append(particle.param_dict.copy())
            else:
                swarm_dicts.append(self._list_to_param_dict(list(particle)))
        
        # Serialize global best
        global_best_dict = None
        if self.global_best is not None:
            if self.global_best.param_dict is not None:
                global_best_dict = self.global_best.param_dict.copy()
            else:
                global_best_dict = self._list_to_param_dict(list(self.global_best))
        
        state.update({
            'swarm_dicts': swarm_dicts,
            'global_best_dict': global_best_dict,
            'global_best_fitness': self.global_best_fitness,
            'iteration': self.iteration,
            'config': self.config
        })
        return state

    def set_state(self, state: dict):
        """Set state from checkpoint."""
        super().set_state(state)
        
        # Ensure toolbox is set up before using it (needed for param_names, etc.)
        if not hasattr(self, 'toolbox') or not hasattr(self, 'param_names'):
            self._setup_toolbox()
        
        # Reconstruct swarm from parameter dicts
        swarm_dicts = state.get('swarm_dicts', [])
        self.swarm = []
        for param_dict in swarm_dicts:
            particle = self._create_particle_deap(param_dict)
            self.swarm.append(particle)
        
        # Reconstruct global best
        global_best_dict = state.get('global_best_dict')
        if global_best_dict is not None:
            self.global_best = self._create_particle_deap(global_best_dict)
            # Evaluate to set fitness
            self.global_best.fitness.values = self.toolbox.evaluate(self.global_best)
            self.global_best_fitness = self.global_best.fitness.values[0]
        else:
            self.global_best = None
            self.global_best_fitness = state.get('global_best_fitness', -float('inf'))
        
        self.iteration = state.get('iteration', 0)
        self.config = state.get('config', self.config)


 
