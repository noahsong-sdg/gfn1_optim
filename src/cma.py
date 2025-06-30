"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for TBLite parameter optimization
Generalized for any molecular system (H2, Si2, etc.)
"""

import numpy as np
import pandas as pd
import toml
import tempfile
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import copy

# External CMA-ES library - install with: pip install cmaes
from cmaes import CMA

# Set up logging with better control
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Control verbosity of external libraries
logging.getLogger('cmaes').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Import project modules
from calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from data_extraction import GFN1ParameterExtractor
from config import get_system_config, SystemConfig

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"


@dataclass
class CMAConfig:
    """Configuration for CMA-ES optimization"""
    sigma: float = 0.5  # Initial step size (standard deviation)
    max_generations: int = 100  # Maximum number of generations
    population_size: Optional[int] = None  # If None, uses CMA-ES default (4 + 3*log(dim))
    seed: Optional[int] = None  # Random seed for reproducibility
    convergence_threshold: float = 1e-6  # Fitness improvement threshold
    patience: int = 20  # Generations without improvement before stopping
    bounds_handling: str = "repair"  # How to handle parameter bounds: "repair" or "penalty"


class GeneralParameterCMA:
    """CMA-ES optimizer for TBLite parameters for any molecular system"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: CMAConfig = CMAConfig(),
                 train_fraction: float = 0.8):
        """Initialize CMA-ES for TBLite parameter optimization
        
        Args:
            system_name: Name of the system to optimize (e.g., 'H2', 'Si2', 'CdS')
            base_param_file: Path to base parameter TOML file
            reference_data: Optional reference data (if None, generates with method)
            config: CMA-ES configuration
            train_fraction: Fraction of data to use for training (rest for testing)
        """
        
        # System configuration
        self.system_name = system_name
        self.system_config = get_system_config(system_name)
        
        # Load base parameters
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        
        # Store base parameter file path for parameter extraction
        self.base_param_file = Path(base_param_file)
        
        # Configuration
        self.config = config
        self.train_fraction = train_fraction
        
        # Define parameter bounds and initial values
        self.parameter_info = self._define_parameter_info()
        
        # Load or generate reference data
        if reference_data is not None:
            self.full_reference_data = reference_data
        else:
            self.full_reference_data = self._load_or_generate_reference_data()
        
        # Split data into train/test sets
        self._split_train_test_data()
        
        # Initialize optimization state
        self.optimizer = None
        self.generation = 0
        self.best_parameters = None
        self.best_fitness = 0.0
        self.convergence_counter = 0
        self.fitness_history = []
        self.failed_evaluations = 0
        
    def _define_parameter_info(self) -> Dict[str, Dict[str, float]]:
        """Define parameter bounds and defaults using extracted defaults"""
        param_info = {}
        
        # Extract default parameters from base parameter file
        extractor = GFN1ParameterExtractor(self.base_param_file)
        system_defaults = extractor.extract_defaults_dict(self.system_config.elements)
        
        # Define bounds based on parameter type and extracted defaults (50% +/- rule)
        for param_name, default_val in system_defaults.items():
            min_val, max_val = self._get_parameter_bounds(param_name, default_val)
            param_info[param_name] = {
                'default': default_val,
                'min': min_val,
                'max': max_val
            }
        
        logger.info(f"Generated {len(param_info)} parameters for {self.system_name} optimization")
        return param_info
    
    def _get_parameter_bounds(self, param_name: str, default_val: float) -> Tuple[float, float]:
        """Get appropriate bounds for a parameter (50% +/- around default)"""
        
        # System-specific parameter bounds (can be extended for different systems)
        system_specific_bounds = {
            'H2': {
                'hamiltonian.xtb.kpair.H-H': (0.5, 1.5),
            },
            'Si2': {
                'hamiltonian.xtb.kpair.Si-Si': (0.5, 1.5),
            },
        }
        
        # Check for system-specific bounds first
        if self.system_name in system_specific_bounds:
            if param_name in system_specific_bounds[self.system_name]:
                return system_specific_bounds[self.system_name][param_name]
        
        # General parameter bounds
        general_bounds = {
            'hamiltonian.xtb.kpol': (1.0, 5.0),
            'hamiltonian.xtb.enscale': (-0.02, 0.02),
            'hamiltonian.xtb.shell.ss': (1.0, 3.0),
            'hamiltonian.xtb.shell.pp': (1.5, 3.5),
            'hamiltonian.xtb.shell.sp': (1.5, 3.0),
        }
        
        if param_name in general_bounds:
            return general_bounds[param_name]
        
        # Type-based bounds
        if 'levels' in param_name:
            # Energy levels - allow ±30% variation
            return (default_val - abs(default_val) * 0.3, default_val + abs(default_val) * 0.3)
        elif 'slater' in param_name:
            # Slater exponents - keep positive with more conservative bounds
            # Ensure minimum is at least 0.5 and not more than 80% below default
            min_safe = max(0.5, default_val * 0.2)  # More conservative minimum
            max_safe = default_val * 1.8  # More conservative maximum
            return (min_safe, max_safe)
        elif 'kcn' in param_name:
            # Coordination number parameters - keep positive, allow wide range
            return (max(0.001, default_val * 0.1), default_val * 5.0)
        elif param_name.endswith('.gam'):
            return (0.2, 0.8)
        elif param_name.endswith('.zeff'):
            return (0.8, 1.5)
        elif param_name.endswith('.arep'):
            return (1.5, 3.0)
        elif param_name.endswith('.en'):
            return (1.5, 3.0)
        else:
            # Default: ±50% around default
            margin = abs(default_val) * 0.5
            return (default_val - margin, default_val + margin)
    
    def _load_or_generate_reference_data(self) -> pd.DataFrame:
        """Load or generate reference data for the system"""
        # For H2, always use CCSD reference data
        if self.system_name == "H2":
            ccsd_file = RESULTS_DIR / "curves" / "h2_ccsd_500.csv"
            if ccsd_file.exists():
                logger.info(f"Loading CCSD reference data from {ccsd_file}")
                return pd.read_csv(ccsd_file)
            else:
                raise FileNotFoundError(f"CCSD reference file {ccsd_file} not found. "
                                      "CMA-ES requires CCSD data for meaningful optimization.")
        
        # For other systems, try system-specific reference file
        ref_file = Path(self.system_config.reference_data_file)
        if ref_file.exists():
            logger.info(f"Loading reference data from {ref_file}")
            return pd.read_csv(ref_file)
        else:
            logger.warning(f"No reference data found for {self.system_name}. "
                          f"Generating with GFN1-xTB (not recommended for optimization).")
            # Generate with GFN1-xTB as fallback
            calc_config = CalcConfig(method=CalcMethod.GFN1_XTB)
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            ref_data = generator.generate_curve(
                save=True, filename=str(ref_file)
            )
            return ref_data
    
    def _split_train_test_data(self):
        """Split reference data into training and test sets using random sampling"""
        # Get distance and energy columns from full dataset
        if 'Distance' in self.full_reference_data.columns:
            full_distances = self.full_reference_data['Distance'].values
            full_energies = self.full_reference_data['Energy'].values
        else:
            # Fallback column names
            full_distances = self.full_reference_data.iloc[:, 0].values
            full_energies = self.full_reference_data.iloc[:, 1].values
        
        # Calculate number of training points
        n_total = len(full_distances)
        n_train = int(n_total * self.train_fraction)
        
        # Use random split for proper statistical validation
        # Set seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split the data
        self.train_distances = full_distances[train_indices]
        self.train_energies = full_energies[train_indices]
        self.test_distances = full_distances[test_indices]
        self.test_energies = full_energies[test_indices]
        
        # Create training reference DataFrame
        self.reference_data = pd.DataFrame({
            'Distance': self.train_distances,
            'Energy': self.train_energies
        })
        
        # Create test reference DataFrame
        self.test_reference_data = pd.DataFrame({
            'Distance': self.test_distances,
            'Energy': self.test_energies
        })
        
        logger.info(f"Data split: {n_train} training points, {len(test_indices)} test points")
        logger.info(f"Training distance range: {self.train_distances.min():.2f} - {self.train_distances.max():.2f} Å")
        logger.info(f"Test distance range: {self.test_distances.min():.2f} - {self.test_distances.max():.2f} Å")
    
    def _set_parameter_in_dict(self, param_dict: dict, path: str, value: float):
        """Set a parameter value using path like 'element.H.levels[0]' or 'hamiltonian.xtb.kpol'"""
        import re
        
        # Convert numpy types to native Python types to avoid TOML serialization issues
        if hasattr(value, 'item'):  # numpy scalar
            value = value.item()
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.integer):
            value = int(value)
        
        # Check if this is an array access
        if '[' in path and ']' in path:
            # Split into path and array index: 'element.H.levels[0]' -> 'element.H.levels', '0'
            match = re.match(r'(.+)\[(\d+)\]$', path)
            if match:
                array_path, index_str = match.groups()
                index = int(index_str)
                
                # Navigate to the array location
                keys = array_path.split('.')
                current = param_dict
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the array element
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
        """Create a temporary parameter file with given parameters"""
        # Start with base parameters
        params = copy.deepcopy(self.base_params)
        
        # Apply given parameters
        for param_name, value in parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(params, f)
            return f.name
    
    def apply_bounds(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply parameter bounds by clamping values"""
        bounded_params = {}
        for param_name, value in parameters.items():
            if param_name in self.parameter_info:
                min_val = self.parameter_info[param_name]['min']
                max_val = self.parameter_info[param_name]['max']
                bounded_value = max(min_val, min(max_val, value))
                
                # Extra safety check for Slater exponents
                if 'slater' in param_name:
                    bounded_value = max(0.5, bounded_value)  # Absolute minimum for safety
                
                bounded_params[param_name] = float(bounded_value)
            else:
                bounded_params[param_name] = float(value)
        return bounded_params
    
    def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
        """Evaluate fitness of parameters by calculating curve error on training data"""
        try:
            # Apply bounds if using repair method
            if self.config.bounds_handling == "repair":
                parameters = self.apply_bounds(parameters)
            
            param_file = self.create_param_file(parameters)
            
            # Create calculator with custom parameters
            calc_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file
            )
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Calculate curve on TRAINING distances only
            calc_data = generator.generate_curve(self.train_distances)
            
            # Clean up temp file
            os.unlink(param_file)
            
            # Compare with TRAINING reference data
            ref_energies = self.reference_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            # Verify shapes match
            if len(ref_energies) != len(calc_energies):
                raise ValueError(f"Shape mismatch: reference {len(ref_energies)} vs calculated {len(calc_energies)}")
            
            # Convert to relative energies
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            # Calculate RMSE on training data
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            
            # For CMA-ES, we minimize the error directly (unlike GA which uses 1/(1+error))
            fitness = rmse
            
            # Apply penalty for out-of-bounds parameters if using penalty method
            if self.config.bounds_handling == "penalty":
                penalty = 0.0
                for param_name, value in parameters.items():
                    if param_name in self.parameter_info:
                        min_val = self.parameter_info[param_name]['min']
                        max_val = self.parameter_info[param_name]['max']
                        if value < min_val or value > max_val:
                            penalty += 1000.0  # Large penalty for constraint violation
                fitness += penalty
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            self.failed_evaluations += 1
            return float('inf')  # Return very large value for failed evaluations
    
    def evaluate_test_performance(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Evaluate parameters' performance on the test set"""
        try:
            bounded_params = self.apply_bounds(parameters)
            param_file = self.create_param_file(bounded_params)
            
            # Create calculator with custom parameters
            calc_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file
            )
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Calculate curve on TEST distances
            calc_data = generator.generate_curve(self.test_reference_data['Distance'].values)
            
            # Clean up temp file
            os.unlink(param_file)
            
            # Compare with TEST reference data
            ref_energies = self.test_reference_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            # Convert to relative energies
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            mae = np.mean(np.abs(ref_relative - calc_relative))
            max_error = np.max(np.abs(ref_relative - calc_relative))
            
            return {
                'test_rmse': rmse,
                'test_mae': mae,
                'test_max_error': max_error
            }
            
        except Exception as e:
            logger.warning(f"Test evaluation failed: {e}")
            return {
                'test_rmse': float('inf'),
                'test_mae': float('inf'),
                'test_max_error': float('inf')
            }
    
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
        """Run CMA-ES optimization"""
        logger.info(f"Starting CMA-ES optimization for {self.system_name}")
        logger.info(f"Using {len(self.train_distances)} training points for optimization")
        start_time = time.time()
        
        # Prepare initial parameters
        param_names = list(self.parameter_info.keys())
        initial_mean = np.array([self.parameter_info[name]['default'] for name in param_names])
        
        logger.info(f"Parameter summary:")
        for i, name in enumerate(param_names[:5]):  # Show first 5 parameters
            pinfo = self.parameter_info[name]
            logger.info(f"  {name}: default={pinfo['default']:.4f}, range=[{pinfo['min']:.4f}, {pinfo['max']:.4f}]")
        if len(param_names) > 5:
            logger.info(f"  ... and {len(param_names)-5} more parameters")
        
        # Initialize CMA-ES optimizer
        try:
            self.optimizer = CMA(
                mean=initial_mean,
                sigma=self.config.sigma,
                population_size=self.config.population_size,
                seed=self.config.seed
            )
            logger.info(f"CMA-ES initialized with population size: {self.optimizer.population_size}")
        except Exception as e:
            logger.error(f"Failed to initialize CMA-ES optimizer: {e}")
            raise RuntimeError(f"CMA-ES initialization failed: {e}") from e
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Log every 10 generations or first/last few
            should_log = (generation % 10 == 0 or generation < 3 or 
                         generation >= self.config.max_generations - 3)
            
            if should_log:
                logger.info(f"Generation {generation + 1}/{self.config.max_generations}")
            
            # Ask for new parameter sets
            solutions = []
            gen_start = time.time()
            
            for _ in range(self.optimizer.population_size):
                # Get parameter vector from CMA-ES
                x = self.optimizer.ask()
                
                # Convert to parameter dictionary (ensure numpy types become Python floats)
                params = {param_names[i]: float(x[i]) for i in range(len(param_names))}
                
                # Evaluate fitness
                fitness = self.evaluate_fitness(params)
                solutions.append((x, fitness))
            
            # Tell CMA-ES the results
            self.optimizer.tell(solutions)
            
            eval_time = time.time() - gen_start
            
            # Track best solution
            best_solution = min(solutions, key=lambda x: x[1])
            best_x, best_fitness = best_solution
            best_params = {param_names[i]: float(best_x[i]) for i in range(len(param_names))}
            
            # Update best if improved
            if self.best_parameters is None or best_fitness < self.best_fitness:
                self.best_parameters = best_params.copy()
                self.best_fitness = best_fitness
                self.convergence_counter = 0
                logger.info(f"  NEW BEST at gen {generation + 1}: RMSE = {best_fitness:.6f}")
            else:
                self.convergence_counter += 1
            
            # Record fitness history
            avg_fitness = np.mean([sol[1] for sol in solutions])
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std([sol[1] for sol in solutions])
            })
            
            # Log progress (reduced frequency)
            if should_log:
                logger.info(f"  Best RMSE: {best_fitness:.6f} | Avg: {avg_fitness:.6f} | "
                           f"Fails: {self.failed_evaluations} | Time: {eval_time:.1f}s")
            
            # Early stopping if all fitness values are inf
            if best_fitness == float('inf'):
                logger.error("All individuals have infinite fitness - stopping optimization")
                break
            
            # Check convergence
            if self.check_convergence():
                logger.info(f"Converged at generation {generation + 1}")
                break
            
            # Check CMA-ES stopping condition
            if self.optimizer.should_stop():
                logger.info(f"CMA-ES stopping criterion met at generation {generation + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        logger.info(f"Total failed evaluations: {self.failed_evaluations}")
        
        # Evaluate best parameters on test set
        if self.best_parameters is not None:
            test_results = self.evaluate_test_performance(self.best_parameters)
            logger.info("Test set performance:")
            logger.info(f"  Test RMSE: {test_results['test_rmse']:.6f}")
            logger.info(f"  Test MAE: {test_results['test_mae']:.6f}")
            logger.info(f"  Test Max Error: {test_results['test_max_error']:.6f}")
        
        return self.best_parameters
    
    def save_best_parameters(self, filename: str):
        """Save the best parameters to a TOML file"""
        if self.best_parameters is None:
            raise ValueError("No optimization has been run")
        
        # Create full parameter set
        params = copy.deepcopy(self.base_params)
        for param_name, value in self.best_parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        with open(filename, 'w') as f:
            toml.dump(params, f)
        
        logger.info(f"Best parameters saved to {filename}")
    
    def save_fitness_history(self, filename: str):
        """Save fitness history to CSV"""
        if not self.fitness_history:
            raise ValueError("No optimization has been run")
        
        df = pd.DataFrame(self.fitness_history)
        df.to_csv(filename, index=False)
        logger.info(f"Fitness history saved to {filename}")


def main():
    """Example usage with different systems"""
    import sys
    
    try:
        # Allow system selection from command line
        if len(sys.argv) > 1:
            system_name = sys.argv[1]
        else:
            system_name = "H2"  # Default
        
        print(f"="*60)
        print(f"CMA-ES Optimization for {system_name}")
        print(f"="*60)
        
        # CMA-ES configuration
        config = CMAConfig(
            sigma=0.5,  # Initial step size
            max_generations=100,
            population_size=None,  # Use CMA-ES default
            convergence_threshold=1e-6,
            patience=20
        )
        
        print(f"Configuration: sigma={config.sigma}, max_gen={config.max_generations}")
        print(f"Base parameter file: {BASE_PARAM_FILE}")
        
        # Initialize CMA-ES
        print(f"\nInitializing CMA-ES optimizer...")
        cma_optimizer = GeneralParameterCMA(
            system_name=system_name,
            base_param_file=str(BASE_PARAM_FILE),
            config=config
        )
        
        print(f"Parameters to optimize: {len(cma_optimizer.parameter_info)}")
        print(f"Training points: {len(cma_optimizer.train_distances)}")
        print(f"Test points: {len(cma_optimizer.test_distances)}")
        
        # Run optimization
        print(f"\nStarting optimization...")
        best_parameters = cma_optimizer.optimize()
        
        # Save results
        print(f"\nSaving results...")
        cma_optimizer.save_best_parameters(cma_optimizer.system_config.optimized_params_file)
        cma_optimizer.save_fitness_history(cma_optimizer.system_config.fitness_history_file)
        
        # Print best parameters
        if best_parameters:
            print(f"\n" + "="*60)
            print(f"OPTIMIZATION COMPLETED SUCCESSFULLY")
            print(f"="*60)
            print(f"Best RMSE: {cma_optimizer.best_fitness:.6f}")
            print(f"Best Parameters for {system_name}:")
            for param_name, value in best_parameters.items():
                print(f"  {param_name}: {value:.6f}")
            print(f"="*60)
        else:
            print(f"\n" + "="*60)
            print(f"OPTIMIZATION FAILED")
            print(f"="*60)
            print("No valid parameters found")
            
    except ImportError as e:
        print(f"\n" + "="*60)
        print(f"IMPORT ERROR")
        print(f"="*60)
        print(f"Missing dependency: {e}")
        print(f"Try: pip install cmaes")
        print(f"="*60)
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"\n" + "="*60)
        print(f"FILE NOT FOUND ERROR")
        print(f"="*60)
        print(f"Missing file: {e}")
        print(f"Make sure you have the required reference data and config files")
        print(f"="*60)
        sys.exit(1)
        
    except Exception as e:
        print(f"\n" + "="*60)
        print(f"UNEXPECTED ERROR")
        print(f"="*60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"="*60)
        sys.exit(1)


if __name__ == "__main__":
    main() 
