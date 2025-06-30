"""
Particle Swarm Optimization for TBLite parameter optimization
Generalized for any molecular system (H2, Si2, etc.)

make sure to change CCSD reference data and the output files and the SPIN variable
"""

import numpy as np
import pandas as pd
import toml
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging - reduce verbosity during optimization
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

# CHANGE THESE -------------------------------------------------
CCSD_REFERENCE_DATA = RESULTS_DIR / "curves" / "si2_ccsd_500.csv" 
PSO_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "si_optim.toml"
PSO_FITNESS_HISTORY = RESULTS_DIR / "fitness" / "si_fitness_history.csv"
SPIN = 2 # numberofalpha electrons - numberofbeta electrons


from data_extraction import GFN1ParameterExtractor
from config import get_system_config

@dataclass
class PSOConfig:
    """Configuration for Particle Swarm Optimization"""
    n_particles: int = 30
    max_iterations: int = 100
    w_max: float = 0.9  # Maximum inertia weight
    w_min: float = 0.4  # Minimum inertia weight
    c1: float = 2.0     # Cognitive acceleration coefficient
    c2: float = 2.0     # Social acceleration coefficient
    max_velocity: float = 0.1  # Maximum velocity as fraction of parameter range
    convergence_threshold: float = 1e-6
    patience: int = 15  # Iterations without improvement before stopping
    max_workers: int = 4

@dataclass
class ParameterBounds:
    """Parameter bounds for PSO optimization"""
    name: str
    min_val: float
    max_val: float
    default_val: float

class Particle:
    """Individual particle in the swarm"""
    
    def __init__(self, parameters: Dict[str, float], bounds: List[ParameterBounds]):
        self.parameters = parameters.copy()
        self.velocity = {name: 0.0 for name in parameters.keys()}
        self.best_parameters = parameters.copy()
        self.best_fitness = float('inf')
        self.current_fitness = float('inf')
        self.bounds = {bound.name: bound for bound in bounds}
        
    def update_velocity(self, global_best_params: Dict[str, float], 
                       w: float, c1: float, c2: float, max_vel: float):
        """Update particle velocity using PSO equations"""
        for param_name in self.parameters.keys():
            r1, r2 = random.random(), random.random()
            
            cognitive = c1 * r1 * (self.best_parameters[param_name] - self.parameters[param_name])
            social = c2 * r2 * (global_best_params[param_name] - self.parameters[param_name])
            
            self.velocity[param_name] = w * self.velocity[param_name] + cognitive + social
            
            # Clamp velocity
            bound = self.bounds[param_name]
            max_velocity = max_vel * (bound.max_val - bound.min_val)
            self.velocity[param_name] = max(-max_velocity, 
                                          min(max_velocity, self.velocity[param_name]))
    
    def update_position(self):
        """Update particle position and apply bounds"""
        for param_name in self.parameters.keys():
            self.parameters[param_name] += self.velocity[param_name]
            
            # Apply bounds
            bound = self.bounds[param_name]
            self.parameters[param_name] = max(bound.min_val, 
                                            min(bound.max_val, self.parameters[param_name]))
    
    def update_best(self):
        """Update personal best if current position is better"""
        if self.current_fitness < self.best_fitness:
            self.best_fitness = self.current_fitness
            self.best_parameters = self.parameters.copy()
            return True
        return False

class TBLiteParameterPSO:
    """PSO optimizer for TBLite parameters for any molecular system"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: PSOConfig = PSOConfig(),
                 train_fraction: float = 0.8):
        """Initialize PSO optimizer
        
        Args:
            system_name: Name of system to optimize (e.g., 'H2', 'Si2', 'C2')
            base_param_file: Path to base parameter TOML file
            reference_data: Optional reference data (if None, loads from system config)
            config: PSO configuration
            train_fraction: Fraction of data to use for training
        """
        
        # System configuration
        self.system_name = system_name
        self.system_config = get_system_config(system_name)
        
        # Load base parameters
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        
        self.config = config
        self.train_fraction = train_fraction
        
        # Define parameter bounds using extraction with 50% +/- bounds
        self.parameter_bounds = self._define_parameter_bounds()
        
        # Load and split reference data
        if reference_data is not None:
            self.full_reference_data = reference_data
        else:
            self.full_reference_data = self._load_reference_data()
        
        self._split_train_test_data()
        
        # Initialize swarm
        self.swarm: List[Particle] = []
        self.global_best_params = {}
        self.global_best_fitness = float('inf')
        self.iteration = 0
        self.convergence_counter = 0
        self.fitness_history = []
        self.failed_evaluations = 0
        
    def _define_parameter_bounds(self) -> List[ParameterBounds]:
        """Define parameter bounds using automatic extraction with 50% +/- margins"""
        bounds = []
        
        # Extract default parameters for this system
        extractor = GFN1ParameterExtractor(Path(BASE_PARAM_FILE))
        system_defaults = extractor.extract_defaults_dict(self.system_config.elements)
        
        # Create bounds with 50% +/- margin for each parameter
        for param_name, default_val in system_defaults.items():
            margin = abs(default_val) * 0.5
            min_val = default_val - margin
            max_val = default_val + margin
            
            # Special handling for parameters that must stay positive
            if 'slater' in param_name or 'kcn' in param_name:
                # For parameters that must be positive, don't force negative defaults positive
                if default_val > 0:
                    min_val = max(0.001, min_val)  # Keep positive only if default is positive
            
            # Validation: ensure max_val > min_val
            if max_val <= min_val:
                logger.warning(f"Invalid bounds for {param_name} (default={default_val:.6f}). Using symmetric bounds around default.")
                # Use symmetric bounds that work
                if default_val >= 0:
                    min_val = default_val * 0.5
                    max_val = default_val * 1.5
                else:
                    min_val = default_val * 1.5  # More negative
                    max_val = default_val * 0.5  # Less negative
                
                # Final check
                if max_val <= min_val:
                    min_val = default_val - 0.1
                    max_val = default_val + 0.1
            
            bounds.append(ParameterBounds(param_name, min_val, max_val, default_val))
        
        logger.info(f"Generated {len(bounds)} parameter bounds for {self.system_name} with 50% +/- margins")
        return bounds
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference data for the system"""
        ref_file = Path(self.system_config.reference_data_file)
        if ref_file.exists():
            logger.info(f"Loading reference data from {ref_file}")
            return pd.read_csv(ref_file)
        else:
            logger.warning(f"Reference file {ref_file} not found for {self.system_name}.")
            # Return empty DataFrame as fallback
            return pd.DataFrame(columns=['Distance', 'Energy'])
    
    def _split_train_test_data(self):
        """Split reference data into training and testing sets"""
        n_train = int(len(self.full_reference_data) * self.train_fraction)
        
        # Use every nth point for training to maintain curve shape
        train_indices = np.linspace(0, len(self.full_reference_data) - 1, n_train, dtype=int)
        test_indices = [i for i in range(len(self.full_reference_data)) if i not in train_indices]
        
        self.train_data = self.full_reference_data.iloc[train_indices].copy()
        self.test_data = self.full_reference_data.iloc[test_indices].copy()
        
        self.train_distances = self.train_data['Distance'].values
        self.test_distances = self.test_data['Distance'].values
        
    def _set_parameter_in_dict(self, param_dict: dict, path: str, value: float):
        """Set a parameter value using dot notation path"""
        import re
        
        # Handle array access like 'element.H.levels[0]'
        if '[' in path and ']' in path:
            match = re.match(r'(.+)\[(\d+)\]$', path)
            if match:
                array_path, index_str = match.groups()
                index = int(index_str)
                
                keys = array_path.split('.')
                current = param_dict
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                array_name = keys[-1]
                if array_name not in current:
                    current[array_name] = []
                while len(current[array_name]) <= index:
                    current[array_name].append(0.0)
                current[array_name][index] = value
                return
        
        # Regular path access
        keys = path.split('.')
        current = param_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def create_param_file(self, parameters: Dict[str, float]) -> str:
        """Create temporary parameter file with given parameters"""
        # Start with base parameters
        params = self.base_params.copy()
        
        # Apply optimized parameters
        for param_name, value in parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(params, f)
            return f.name
    
    def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
        """Evaluate fitness of parameter set using system dissociation curve"""
        try:
            param_file = self.create_param_file(parameters)
            
            # Import here to avoid circular imports
            from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
            
            # Create calculator with custom parameters
            custom_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file,
                spin=SPIN
            )
            
            calculator = GeneralCalculator(custom_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Generate curve for training points
            calc_data = generator.generate_curve(distances=self.train_distances, save=False)
            
            # Calculate RMSE against reference
            ref_energies = self.train_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            # Align curves by subtracting minima
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            
            # Clean up temporary file
            Path(param_file).unlink(missing_ok=True)
            
            return rmse
            
        except Exception as e:
            self.failed_evaluations += 1
            return float('inf')
    
    def evaluate_test_performance(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Evaluate performance on test set"""
        try:
            param_file = self.create_param_file(parameters)
            
            from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
            
            custom_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file,
                spin=SPIN
            )
            
            calculator = GeneralCalculator(custom_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Generate test curve
            calc_data = generator.generate_curve(distances=self.test_distances, save=False)
            
            # Calculate metrics
            ref_energies = self.test_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            mae = np.mean(np.abs(ref_relative - calc_relative))
            max_error = np.max(np.abs(ref_relative - calc_relative))
            
            Path(param_file).unlink(missing_ok=True)
            
            return {
                'test_rmse': rmse,
                'test_mae': mae,
                'test_max_error': max_error
            }
            
        except Exception as e:
            return {
                'test_rmse': float('inf'),
                'test_mae': float('inf'),
                'test_max_error': float('inf')
            }
    
    def initialize_swarm(self):
        """Initialize particle swarm with random positions"""
        
        self.swarm = []
        for i in range(self.config.n_particles):
            # Generate random parameters within bounds
            parameters = {}
            for bound in self.parameter_bounds:
                if random.random() < 0.7:  # 70% chance to start near default
                    std = (bound.max_val - bound.min_val) * 0.15
                    value = np.random.normal(bound.default_val, std)
                else:
                    value = random.uniform(bound.min_val, bound.max_val)
                
                # Clamp to bounds
                value = max(bound.min_val, min(bound.max_val, value))
                parameters[bound.name] = value
            
            particle = Particle(parameters, self.parameter_bounds)
            self.swarm.append(particle)
    
    def evaluate_swarm_parallel(self):
        """Evaluate fitness of all particles (serial evaluation to avoid multiprocessing issues)"""
        
        # Note: Using serial evaluation for simplicity and debugging
        for particle in self.swarm:
            try:
                fitness = self.evaluate_fitness(particle.parameters)
                particle.current_fitness = fitness
                
                # Update personal best
                if particle.update_best():
                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_params = particle.parameters.copy()
                        print(f"New global best fitness: {fitness:.6f}")
                        
            except Exception as e:
                logger.error(f"Particle evaluation failed: {e}")
                particle.current_fitness = float('inf')
    
    def update_swarm(self):
        """Update particle velocities and positions"""
        # Calculate inertia weight (linearly decreasing)
        w = self.config.w_max - (self.config.w_max - self.config.w_min) * \
            (self.iteration / self.config.max_iterations)
        
        for particle in self.swarm:
            particle.update_velocity(
                self.global_best_params, w, 
                self.config.c1, self.config.c2, 
                self.config.max_velocity
            )
            particle.update_position()
    
    def check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if len(self.fitness_history) < 2:
            return False
        
        # Check for improvement
        if len(self.fitness_history) >= 2:
            improvement = abs(self.fitness_history[-2] - self.fitness_history[-1])
            if improvement < self.config.convergence_threshold:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0
        
        return self.convergence_counter >= self.config.patience
    
    def optimize(self) -> Dict[str, float]:
        """Run PSO optimization"""
        print(f"Starting PSO optimization for {self.system_name}: {len(self.train_distances)} training points, {self.config.n_particles} particles")
        
        # Initialize swarm
        self.initialize_swarm()
        
        # Initial evaluation
        self.evaluate_swarm_parallel()
        self.fitness_history.append(self.global_best_fitness)
        
        # Main optimization loop
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Show progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.config.max_iterations}, Best: {self.global_best_fitness:.6f}")
            
            # Update swarm
            self.update_swarm()
            
            # Evaluate new positions
            self.evaluate_swarm_parallel()
            self.fitness_history.append(self.global_best_fitness)
            
            # Check convergence
            if self.check_convergence():
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Check for systematic failures
            if self.failed_evaluations > len(self.swarm) * 0.5:
                print("Too many failed evaluations - stopping optimization")
                break
        print(f"Final best fitness: {self.global_best_fitness:.6f}")
        
        return self.global_best_params
    
    def get_best_parameters(self) -> Dict[str, float]:
        """Get the best parameters found"""
        if not self.global_best_params:
            raise ValueError("No optimization has been run")
        return self.global_best_params.copy()
    
    def save_best_parameters(self, filename: str):
        """Save best parameters to TOML file"""
        if not self.global_best_params:
            raise ValueError("No optimization has been run")
        
        # Create full parameter file
        params = self.base_params.copy()
        for param_name, value in self.global_best_params.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        with open(filename, 'w') as f:
            toml.dump(params, f)
    
    def save_fitness_history(self, filename: str):
        """Save fitness history to CSV"""
        df = pd.DataFrame({
            'iteration': range(len(self.fitness_history)),
            'best_fitness': self.fitness_history
        })
        df.to_csv(filename, index=False)

def main():
    """System-agnostic PSO configuration"""
    import sys
    
    # Allow system selection from command line
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
    else:
        system_name = "H2"  # Default
    
    print(f"Running PSO optimization for {system_name}")
    
    # Scaled configuration - can be tuned per system
    pso_config = PSOConfig(
        n_particles=30,      # Enough to explore parameter space efficiently  
        max_iterations=100,   # Should be sufficient for most systems
        max_workers=12
    )
    reference_data = pd.read_csv(CCSD_REFERENCE_DATA)

    # Initialize optimizer
    optimizer = TBLiteParameterPSO(
        system_name=system_name,
        base_param_file=str(BASE_PARAM_FILE),
        reference_data=reference_data,
        config=pso_config
    )
    
    # Run optimization
    best_params = optimizer.optimize()
    
    # Save results with system-specific filenames
    pso_params_file = RESULTS_DIR / "parameters" / f"pso_optimized_params_{system_name.lower()}.toml"
    pso_history_file = RESULTS_DIR / "fitness" / f"pso_fitness_history_{system_name.lower()}.csv"
    
    optimizer.save_best_parameters(str(pso_params_file))
    optimizer.save_fitness_history(str(pso_history_file))
    
    # Evaluate test performance
    test_metrics = optimizer.evaluate_test_performance(best_params)
    print(f"Test Performance for {system_name}:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

if __name__ == "__main__":
    main() 
