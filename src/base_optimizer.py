"""
Base optimizer class for TBLite parameter optimization.
Consolidates common functionality across GA, PSO, Bayesian, and CMA-ES optimizers.
"""

import numpy as np
import pandas as pd
import toml
import tempfile
import logging
import os
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
import pickle
import random

# Import project modules
from calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from data_extraction import GFN1ParameterExtractor, extract_si2_parameters
from config import get_system_config, SystemConfig, CalculationType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Portable paths
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

@dataclass
class ParameterBounds:
    """Bounds for parameter optimization"""
    name: str
    min_val: float
    max_val: float
    default_val: float

@dataclass
class BaseConfig:
    """Base configuration for all optimizers"""
    convergence_threshold: float = 1e-6
    patience: int = 20
    max_workers: int = 4


class BaseOptimizer(ABC):
    """Base class for all TBLite parameter optimizers"""
    
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 train_fraction: float = 0.8,
                 spin: int = 0,
                 method_name: Optional[str] = None):
        """Initialize base optimizer
        
        Args:
            system_name: Name of the system to optimize (e.g., 'H2', 'Si2', 'CdS')
            base_param_file: Path to base parameter TOML file
            reference_data: Optional reference data (if None, loads from system config)
            train_fraction: Fraction of data to use for training (rest for testing)
            spin: Spin multiplicity for calculations
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
        self.train_fraction = train_fraction
        self.spin = spin
        
        # Define parameter bounds using parameter extraction
        self.parameter_bounds = self._define_parameter_bounds()
        
        # Load or generate reference data
        if reference_data is not None:
            self.full_reference_data = reference_data
        else:
            self.full_reference_data = self._load_or_generate_reference_data()
        
        # Split data into train/test sets
        self._split_train_test_data()
        
        # Initialize optimization state
        self.best_parameters = None
        self.best_fitness = float('inf')  # Initialize to infinity for minimization (RMSE)
        self.convergence_counter = 0
        self.fitness_history = []
        self.failed_evaluations = 0
        
        self.method_name = method_name or self.__class__.__name__.replace('GeneralParameter', '').replace('BaseOptimizer', '').lower()
        
        # Check for checkpoint and load if present
        checkpoint_path = self.get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            logger.info(f"Checkpoint found at {checkpoint_path}, loading state...")
            self.load_checkpoint()
        
    def _define_parameter_bounds(self) -> List[ParameterBounds]:
        """Define parameter bounds for system-relevant parameters using extracted defaults"""
        bounds = []
        
        # Use focused parameter set for Si2 to avoid over-parameterization
        if self.system_name == "Si2":
            system_defaults = extract_si2_parameters()
            logger.info(f"Using focused Si2 parameter set with {len(system_defaults)} parameters")
        else:
            # Extract default parameters from base parameter file for other systems
            extractor = GFN1ParameterExtractor(self.base_param_file)
            system_defaults = extractor.extract_defaults_dict(self.system_config.elements)
        
        # Define bounds based on parameter type and extracted defaults
        for param_name, default_val in system_defaults.items():
            min_val, max_val = self._get_parameter_bounds(param_name, default_val)
            bounds.append(ParameterBounds(param_name, min_val, max_val, default_val))
        
        logger.info(f"Generated {len(bounds)} parameter bounds for {self.system_name} from extracted defaults")
        return bounds
    
    def _get_parameter_bounds(self, param_name: str, default_val: float) -> Tuple[float, float]:
        """Get appropriate bounds for a parameter based on its name and default value"""
        # System-specific parameter bounds (can be extended for different systems)
        system_specific_bounds = {
            'H2': {
                'hamiltonian.xtb.kpair.H-H': (0.5, 1.5),
            },
            'Si2': {
                'hamiltonian.xtb.kpair.Si-Si': (0.8, 1.2),  # Tighter bounds, prevent negative values
                'hamiltonian.xtb.kpol': (2.5, 3.5),        # Tighter around default ~2.85
                'hamiltonian.xtb.enscale': (-0.015, 0.002), # Tighter around default -0.007
                'element.Si.gam': (0.35, 0.55),             # Tighter around default 0.438
                'element.Si.zeff': (15.5, 18.5),            # Tighter around default 16.9
                'element.Si.arep': (0.8, 1.1),              # Tighter around default 0.948
                'element.Si.en': (1.7, 2.1),                # Tighter around default 1.9
            },
            'CdS': {
                'hamiltonian.xtb.kpair.Cd-S': (0.7, 1.3),
                'hamiltonian.xtb.kpair.Cd-Cd': (0.7, 1.3),
                'hamiltonian.xtb.kpair.S-S': (0.7, 1.3),
                'hamiltonian.xtb.kpol': (2.5, 3.5),
                'hamiltonian.xtb.enscale': (-0.015, 0.002),
                'element.Cd.gam': (0.35, 0.55),
                'element.Cd.zeff': (15.5, 18.5),
                'element.Cd.arep': (0.8, 1.1),
                'element.Cd.en': (1.7, 2.1),
                'element.S.gam': (0.35, 0.55),
                'element.S.zeff': (15.5, 18.5),
                'element.S.arep': (0.8, 1.1),
                'element.S.en': (1.7, 2.1),
            },
        }
        # Check for system-specific bounds first
        if self.system_name in system_specific_bounds:
            if param_name in system_specific_bounds[self.system_name]:
                return system_specific_bounds[self.system_name][param_name]
        # General parameter bounds
        general_bounds = {
            'hamiltonian.xtb.kpol': (2.0, 3.5),      # Tighter lower bound
            'hamiltonian.xtb.enscale': (-0.015, 0.002), # Tighter upper bound
            'hamiltonian.xtb.shell.ss': (1.0, 2.0),
            'hamiltonian.xtb.shell.pp': (1.5, 2.5),
            'hamiltonian.xtb.shell.sp': (1.5, 2.5),
        }
        if param_name in general_bounds:
            return general_bounds[param_name]
        # Type-based bounds
        if 'levels' in param_name:
            # Energy levels - must be negative, allow ±20% variation, min -20.0, max -0.1
            margin = abs(default_val) * 0.2
            min_val = min(-0.1, default_val - margin)
            max_val = max(-20.0, default_val + margin)
            if min_val > -0.1:
                min_val = -0.1
            if max_val < -20.0:
                max_val = -20.0
            return (max_val, min_val) if max_val < min_val else (min_val, max_val)
        elif 'slater' in param_name:
            # Slater exponents - must be positive, typical range 0.5–2.0
            min_val = max(0.5, default_val * 0.8)
            max_val = min(2.0, default_val * 1.2)
            return (min_val, max_val)
        elif 'kcn' in param_name:
            # Coordination number parameters - must be positive, typical range 0.01–1.0
            min_val = max(0.01, default_val * 0.5)
            max_val = min(1.0, default_val * 1.5)
            return (min_val, max_val)
        elif 'kpair' in param_name:
            # Pair parameters - must be positive, typical range 0.5–1.5
            min_val = max(0.5, default_val * 0.8)
            max_val = min(1.5, default_val * 1.2)
            return (min_val, max_val)
        elif param_name.endswith('.gam'):
            # Gamma parameters - typical range 0.3–0.6
            return (0.3, 0.6)
        elif param_name.endswith('.zeff'):
            # Effective charge - typical range 10.0–20.0
            return (10.0, 20.0)
        elif param_name.endswith('.arep'):
            # Repulsion parameters - typical range 0.8–1.2
            return (0.8, 1.2)
        elif param_name.endswith('.en'):
            # Electronegativity - typical range 1.5–2.5
            return (1.5, 2.5)
        else:
            # Default: ±15% around default, but never allow negative for physical parameters
            margin = abs(default_val) * 0.15
            min_val = default_val - margin
            max_val = default_val + margin
            if min_val < 0 and default_val > 0:
                min_val = 0.01
            if max_val <= min_val:
                return (default_val - 0.05, default_val + 0.05)
            return (min_val, max_val)
    
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
                                      "Optimization requires CCSD data for meaningful results.")
        elif self.system_name == "Si2":
            ccsd_file = RESULTS_DIR / "curves" / "si2_ccsd_500.csv"
            if ccsd_file.exists():
                return pd.read_csv(ccsd_file)
        
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
            # Find the corresponding bound
            bound = next((b for b in self.parameter_bounds if b.name == param_name), None)
            if bound:
                bounded_value = max(bound.min_val, min(bound.max_val, value))
                
                # Extra safety checks for critical parameters
                if 'slater' in param_name:
                    bounded_value = max(0.5, bounded_value)  # Absolute minimum for safety
                elif 'zeff' in param_name:
                    bounded_value = max(1.0, bounded_value)  # Effective charge must be positive
                elif 'kcn' in param_name and param_name.endswith('[0]'):
                    bounded_value = max(0.01, bounded_value)  # First kcn parameter must be positive
                elif 'kpair' in param_name:
                    bounded_value = max(0.1, bounded_value)  # Pair parameters must be positive
                elif 'gam' in param_name and not param_name.endswith('lgam'):
                    bounded_value = max(0.1, min(1.0, bounded_value))  # Gamma parameters reasonable range
                elif 'arep' in param_name:
                    bounded_value = max(0.5, bounded_value)  # Repulsion parameters must be positive
                elif 'en' in param_name and not param_name.endswith('zen'):
                    bounded_value = max(0.5, bounded_value)  # Electronegativity must be positive
                
                bounded_params[param_name] = float(bounded_value)
            else:
                bounded_params[param_name] = float(value)
        return bounded_params
    
    def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
        """Evaluate fitness of parameters by calculating curve error on training data or lattice constants for solids"""
        try:
            # Apply bounds
            parameters = self.apply_bounds(parameters)
            param_file = self.create_param_file(parameters)

            # For lattice constant fitting
            if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
                from calc import GeneralCalculator, LatticeConstantsGenerator, CalcConfig, CalcMethod
                calc_config = CalcConfig(
                    method=CalcMethod.XTB_CUSTOM,
                    param_file=param_file,
                    spin=self.spin
                )
                calculator = GeneralCalculator(calc_config, self.system_config)
                generator = LatticeConstantsGenerator(calculator)
                # Optimize lattice constants
                a_opt, c_opt, _ = generator.optimize_lattice_constants()
                # Reference values
                a_ref = self.system_config.lattice_params["a"]
                c_ref = self.system_config.lattice_params["c"]
                # Loss: squared error
                loss = (a_opt - a_ref) ** 2 + (c_opt - c_ref) ** 2
                # Clean up temp file
                os.unlink(param_file)
                return loss

            # Default: molecular (curve) fitting
            # Create calculator with custom parameters
            calc_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file,
                spin=self.spin
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
            
            return rmse
            
        except Exception as e:
            # Log the specific parameters that caused the failure for debugging
            param_summary = {k: f"{v:.6f}" for k, v in parameters.items()}
            logger.warning(f"Fitness evaluation failed with parameters {param_summary}: {e}")
            self.failed_evaluations += 1
            
            # Return a large but finite penalty instead of infinity
            return 1000.0
    
    def evaluate_test_performance(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Evaluate parameters' performance on the test set or lattice constants for solids"""
        try:
            bounded_params = self.apply_bounds(parameters)
            param_file = self.create_param_file(bounded_params)

            if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
                from calc import GeneralCalculator, LatticeConstantsGenerator, CalcConfig, CalcMethod
                calc_config = CalcConfig(
                    method=CalcMethod.XTB_CUSTOM,
                    param_file=param_file,
                    spin=self.spin
                )
                calculator = GeneralCalculator(calc_config, self.system_config)
                generator = LatticeConstantsGenerator(calculator)
                a_opt, c_opt, _ = generator.optimize_lattice_constants()
                a_ref = self.system_config.lattice_params["a"]
                c_ref = self.system_config.lattice_params["c"]
                a_error = abs(a_opt - a_ref)
                c_error = abs(c_opt - c_ref)
                os.unlink(param_file)
                return {
                    'test_a_error': a_error,
                    'test_c_error': c_error,
                    'test_total_error': a_error + c_error
                }

            # Default: molecular (curve) test evaluation
            # Create calculator with custom parameters
            calc_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file,
                spin=self.spin
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
                'test_rmse': 1000.0,
                'test_mae': 1000.0,
                'test_max_error': 1000.0
            }
    
    def check_convergence(self) -> bool:
        """Check if the algorithm has converged (basic implementation)"""
        if len(self.fitness_history) < 2:
            return False
        
        # This is a basic implementation - subclasses can override
        # Check if improvement is below threshold
        if hasattr(self, 'config') and hasattr(self.config, 'convergence_threshold'):
            threshold = self.config.convergence_threshold
            patience = getattr(self.config, 'patience', 20)
        else:
            threshold = 1e-6
            patience = 20
        
        recent_improvement = abs(
            self.fitness_history[-2] - self.fitness_history[-1]
        )
        
        if recent_improvement < threshold:
            self.convergence_counter += 1
            return self.convergence_counter >= patience
        else:
            self.convergence_counter = 0
            return False
    
    def get_method_specific_filename(self, base_filename: str) -> str:
        """Generate a method-specific filename by inserting the method name before the extension"""
        path = Path(base_filename)
        # Insert method name before extension: si2_optimized.toml -> si2_bayes.toml
        new_stem = f"{path.stem}_{self.method_name}"
        return str(path.parent / f"{new_stem}{path.suffix}")
    
    def get_optimized_params_filename(self) -> str:
        """Get the method-specific optimized parameters filename"""
        return self.get_method_specific_filename(self.system_config.optimized_params_file)
    
    def get_fitness_history_filename(self) -> str:
        """Get the method-specific fitness history filename"""
        return self.get_method_specific_filename(self.system_config.fitness_history_file)
    
    def save_best_parameters(self, filename: Optional[str] = None):
        """Save the best parameters to a TOML file"""
        if self.best_parameters is None:
            raise ValueError("No optimization has been run")
        
        # Use method-specific filename if none provided
        if filename is None:
            filename = self.get_optimized_params_filename()
        
        # Create full parameter set
        params = copy.deepcopy(self.base_params)
        for param_name, value in self.best_parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            toml.dump(params, f)
        
        logger.info(f"Best parameters saved to {filename}")
    
    def save_fitness_history(self, filename: Optional[str] = None):
        """Save fitness history to CSV"""
        if not self.fitness_history:
            raise ValueError("No optimization has been run")
        
        # Use method-specific filename if none provided
        if filename is None:
            filename = self.get_fitness_history_filename()
        
        # Handle different fitness history formats
        if isinstance(self.fitness_history[0], dict):
            # For algorithms that store dict with generation info
            df = pd.DataFrame(self.fitness_history)
        else:
            # For algorithms that store just fitness values
            df = pd.DataFrame({
                'iteration': range(len(self.fitness_history)),
                'best_fitness': self.fitness_history
            })
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filename, index=False)
        logger.info(f"Fitness history saved to {filename}")
    
    def get_checkpoint_path(self) -> str:
        """Return the checkpoint filename for this system/method."""
        return f"{self.system_name.lower()}_{self.method_name.lower()}_checkpt.pkl"

    def save_checkpoint(self):
        """Save minimal optimizer state to a checkpoint file (overwrites each time)."""
        try:
            state = self.get_state()
            # Save RNG state
            state['random_state'] = random.getstate()
            state['np_random_state'] = np.random.get_state()
            with open(self.get_checkpoint_path(), 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Checkpoint saved to {self.get_checkpoint_path()}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            logger.info("Continuing without checkpoint save")

    def load_checkpoint(self):
        """Load optimizer state from checkpoint file."""
        try:
            with open(self.get_checkpoint_path(), 'rb') as f:
                state = pickle.load(f)
            self.set_state(state)
            # Restore RNG state
            if 'random_state' in state:
                random.setstate(state['random_state'])
            if 'np_random_state' in state:
                np.random.set_state(state['np_random_state'])
            logger.info(f"Checkpoint loaded from {self.get_checkpoint_path()}")
        except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
            logger.warning(f"Failed to load checkpoint from {self.get_checkpoint_path()}: {e}")
            logger.info("Starting fresh optimization")
            # Reset state to fresh start
            self.best_parameters = None
            self.best_fitness = float('inf')
            self.convergence_counter = 0
            self.fitness_history = []
            self.failed_evaluations = 0

    @abstractmethod
    def optimize(self) -> Dict[str, float]:
        """Run the optimization algorithm - must be implemented by subclasses"""
        pass
    
    def get_state(self) -> dict:
        """Return a dict of minimal state for checkpointing (base fields only)"""
        return {
            'best_parameters': self.best_parameters,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'convergence_counter': getattr(self, 'convergence_counter', 0)
        }

    def set_state(self, state: dict):
        """Restore state from dict (base fields only)"""
        self.best_parameters = state.get('best_parameters')
        self.best_fitness = state.get('best_fitness', float('inf'))
        self.fitness_history = state.get('fitness_history', [])
        self.convergence_counter = state.get('convergence_counter', 0)
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names being optimized"""
        return [bound.name for bound in self.parameter_bounds]
    
    def get_parameter_defaults(self) -> Dict[str, float]:
        """Get default parameter values"""
        return {bound.name: bound.default_val for bound in self.parameter_bounds}
    
    def get_parameter_bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds as dictionary"""
        return {bound.name: (bound.min_val, bound.max_val) for bound in self.parameter_bounds} 
