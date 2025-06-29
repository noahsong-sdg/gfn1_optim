"""
Bayesian Optimization (Gaussian Process) for TBLite parameter optimization
"""

import numpy as np
import pandas as pd
import toml
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Bayesian optimization imports
from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
from skopt import dump, load

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

# Reference data files
CCSD_REFERENCE_DATA = RESULTS_DIR / "curves" / "h2_ccsd_500.csv"

# Output files
BAYES_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "bayes_optimized_params_v2.toml"
BAYES_FITNESS_HISTORY = RESULTS_DIR / "fitness" / "bayes_fitness_history_v2.csv"
BAYES_RESULTS_PICKLE = RESULTS_DIR / "parameters" / "bayes_results_v2.pkl"

@dataclass
class BayesConfig:
    """Configuration for Bayesian Optimization"""
    n_calls: int = 100               # Number of function evaluations
    n_initial_points: int = 20       # Random exploration before GP
    n_restarts_optimizer: int = 10   # GP hyperparameter optimization restarts
    acquisition_func: str = "EI"     # EI, PI, LCB
    acquisition_weight: float = 0.01 # For LCB: lower = more exploration
    random_state: int = 42
    convergence_threshold: float = 1e-6
    patience: int = 15               # Calls without improvement before stopping
    noise: float = 1e-10            # GP noise variance
    
@dataclass
class ParameterSpace:
    """Parameter space definition for Bayesian optimization"""
    name: str
    min_val: float
    max_val: float
    default_val: float

class TBLiteParameterBayesian:
    """Bayesian optimizer for TBLite parameters using H2 dissociation data"""
    
    def __init__(self, 
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: BayesConfig = BayesConfig(),
                 train_fraction: float = 0.8):
        """Initialize Bayesian optimizer"""
        
        # Load base parameters
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        
        self.config = config
        self.train_fraction = train_fraction
        
        # Define H2-relevant parameter space
        self.parameter_space = self._define_h2_parameter_space()
        self.param_names = [param.name for param in self.parameter_space]
        
        # Create scikit-optimize dimensions
        self.dimensions = [
            Real(param.min_val, param.max_val, name=param.name, prior='uniform')
            for param in self.parameter_space
        ]
        
        # Load and split reference data
        if reference_data is not None:
            self.full_reference_data = reference_data
        else:
            self.full_reference_data = self._load_reference_data()
        
        self._split_train_test_data()
        
        # Optimization tracking
        self.iteration = 0
        self.best_fitness = float('inf')
        self.best_params = {}
        self.fitness_history = []
        self.convergence_counter = 0
        self.failed_evaluations = 0
        
    def _define_h2_parameter_space(self) -> List[ParameterSpace]:
        """Define parameter space for H2-relevant parameters"""
        space = []
        
        # Hamiltonian parameters
        space.extend([
            ParameterSpace("hamiltonian.xtb.kpol", 1.0, 5.0, 2.85),
            ParameterSpace("hamiltonian.xtb.enscale", -0.02, 0.02, -0.007),
        ])
        
        # Shell parameters
        space.extend([
            ParameterSpace("hamiltonian.xtb.shell.ss", 1.0, 3.0, 1.85),
            ParameterSpace("hamiltonian.xtb.shell.pp", 1.5, 3.5, 2.25),
            ParameterSpace("hamiltonian.xtb.shell.sp", 1.5, 3.0, 2.08),
        ])
        
        # H-H pair interaction
        space.append(ParameterSpace("hamiltonian.xtb.kpair.H-H", 0.5, 1.5, 0.96))
        
        # Hydrogen element parameters
        h_element_space = [
            ("element.H.levels[0]", -15.0, -8.0, -10.92),  # 1s level
            ("element.H.levels[1]", -4.0, -1.0, -2.17),   # 2s level
            ("element.H.slater[0]", 0.8, 2.0, 1.21),      # 1s slater
            ("element.H.slater[1]", 1.0, 3.0, 1.99),      # 2s slater
            ("element.H.kcn[0]", 0.01, 0.15, 0.0655),     # coordination number dependence
            ("element.H.kcn[1]", 0.001, 0.05, 0.0130),
            ("element.H.gam", 0.2, 0.8, 0.47),           # gamma parameter
            ("element.H.zeff", 0.8, 1.5, 1.12),          # effective nuclear charge
            ("element.H.arep", 1.5, 3.0, 2.21),          # repulsion parameter
            ("element.H.en", 1.5, 3.0, 2.2),             # electronegativity
        ]
        
        for param_path, min_val, max_val, default in h_element_space:
            space.append(ParameterSpace(param_path, min_val, max_val, default))
            
        return space
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load or generate reference CCSD data for H2"""
        if CCSD_REFERENCE_DATA.exists():
            return pd.read_csv(CCSD_REFERENCE_DATA)
        else:
            logger.warning(f"Reference file {CCSD_REFERENCE_DATA} not found. Please ensure h2_ccsd_data.csv exists.")
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
    
    def create_param_file(self, parameter_values: List[float]) -> str:
        """Create temporary parameter file with given parameter values"""
        # Start with base parameters
        params = self.base_params.copy()
        
        # Apply optimized parameters
        parameter_dict = dict(zip(self.param_names, parameter_values))
        for param_name, value in parameter_dict.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(params, f)
            return f.name
    
    def objective_function(self, parameter_values) -> float:
        """Objective function for Bayesian optimization"""
        try:
            
            self.iteration += 1
            
            param_file = self.create_param_file(parameter_values)
            
            # Import here to avoid circular imports
            from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
            from config import get_system_config
            
            # Create calculator with custom parameters
            custom_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file,
                spin=1
            )
            
            system_config = get_system_config("H2")
            calculator = GeneralCalculator(custom_config, system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Generate H2 curve for training points
            calc_data = generator.generate_curve(distances=self.train_distances, save=False)
            
            # Calculate RMSE against reference
            ref_energies = self.train_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            # Align curves by subtracting minima
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            
            # Update best if improved
            if rmse < self.best_fitness:
                self.best_fitness = rmse
                self.best_params = dict(zip(self.param_names, parameter_values))
                print(f"Evaluation {self.iteration}: New best fitness: {rmse:.6f}")
                self.convergence_counter = 0
            else:
                self.convergence_counter += 1
            
            self.fitness_history.append(rmse)
            
            # Clean up temporary file
            Path(param_file).unlink(missing_ok=True)
            
            return rmse
            
        except Exception as e:
            self.failed_evaluations += 1
            return float('inf')
    
    def evaluate_test_performance(self, parameter_values: List[float]) -> Dict[str, float]:
        """Evaluate performance on test set"""
        try:
            param_file = self.create_param_file(parameter_values)
            
            from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
            from config import get_system_config
            
            custom_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file,
                spin=1
            )
            
            system_config = get_system_config("H2")
            calculator = GeneralCalculator(custom_config, system_config)
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
    
    def optimize(self) -> Dict[str, float]:
        """Run Bayesian optimization"""
        print(f"Starting Bayesian optimization: {len(self.train_distances)} training points, {len(self.dimensions)} parameters")
        
        start_time = time.time()
        
        # Choose acquisition function
        acq_func = self.config.acquisition_func.upper()
        if acq_func == "EI":
            acquisition_function = "EI"
        elif acq_func == "PI":
            acquisition_function = "PI"
        elif acq_func == "LCB":
            acquisition_function = "LCB"
        else:
            logger.warning(f"Unknown acquisition function {acq_func}, using EI")
            acquisition_function = "EI"
        
        # Initial points - start some near defaults
        x0 = []
        y0 = []
        
        # Add default parameters as starting point
        default_values = [param.default_val for param in self.parameter_space]
        print("Evaluating default parameters...")
        default_fitness = self.objective_function(default_values)
        x0.append(default_values)
        y0.append(default_fitness)
        
        # Run Bayesian optimization
        print(f"Running GP optimization with {acquisition_function} acquisition...")
        
        # Create wrapper function for gp_minimize (it expects a function that takes a list)
        def objective_wrapper(params):
            return self.objective_function(params)
        
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.dimensions,
            n_calls=self.config.n_calls,
            n_initial_points=self.config.n_initial_points - 1,  # -1 because we added default
            x0=x0,
            y0=y0,
            acq_func=acquisition_function,
            acq_optimizer="auto",
            n_restarts_optimizer=self.config.n_restarts_optimizer,
            noise=self.config.noise,
            random_state=self.config.random_state,
            callback=self._callback,
            model_queue_size=1  # Don't keep all models in memory
        )
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f}s")
        print(f"Final best fitness: {result.fun:.6f}")
        print(f"Total function evaluations: {len(result.func_vals)}")
        
        # Store final results
        self.best_params = dict(zip(self.param_names, result.x))
        self.best_fitness = result.fun
        self.optimization_result = result
        
        return self.best_params
    
    def _callback(self, result):
        """Callback function for optimization progress"""
        # Check convergence - need at least 2*patience evaluations
        if len(result.func_vals) >= 2 * self.config.patience:
            recent_best = min(result.func_vals[-self.config.patience:])
            older_best = min(result.func_vals[-2*self.config.patience:-self.config.patience])
            improvement = abs(older_best - recent_best)
            if improvement < self.config.convergence_threshold:
                print(f"Converged after {len(result.func_vals)} evaluations")
                return True  # Stop optimization
        
        # Check for too many failures
        if self.failed_evaluations > self.config.n_calls * 0.3:
            print("Too many failed evaluations - stopping optimization")
            return True
            
        return False
    
    def get_best_parameters(self) -> Dict[str, float]:
        """Get the best parameters found"""
        if not self.best_params:
            raise ValueError("No optimization has been run")
        return self.best_params.copy()
    
    def save_best_parameters(self, filename: str):
        """Save best parameters to TOML file"""
        if not self.best_params:
            raise ValueError("No optimization has been run")
        
        # Create full parameter file
        params = self.base_params.copy()
        for param_name, value in self.best_params.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            toml.dump(params, f)
    
    def save_fitness_history(self, filename: str):
        """Save fitness history to CSV"""
        df = pd.DataFrame({
            'evaluation': range(1, len(self.fitness_history) + 1),
            'best_fitness': self.fitness_history
        })
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filename, index=False)
    
    def save_optimization_result(self, filename: str):
        """Save complete optimization result for later analysis"""
        if not hasattr(self, 'optimization_result'):
            raise ValueError("No optimization has been run")
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        dump(self.optimization_result, filename)

def main():
    """H2-optimized Bayesian optimization"""
    # Configuration for H2 - balance exploration vs exploitation
    bayes_config = BayesConfig(
        n_calls=100,                    # Total function evaluations
        n_initial_points=20,            # Random exploration phase
        n_restarts_optimizer=10,        # GP hyperparameter optimization
        acquisition_func="EI",          # Expected Improvement
        random_state=42                 # Reproducibility
    )
    
    # Initialize optimizer
    optimizer = TBLiteParameterBayesian(
        base_param_file=str(BASE_PARAM_FILE),
        config=bayes_config
    )
    
    # Run optimization
    best_params = optimizer.optimize()
    
    # Save results
    optimizer.save_best_parameters(str(BAYES_OPTIMIZED_PARAMS))
    optimizer.save_fitness_history(str(BAYES_FITNESS_HISTORY))
    optimizer.save_optimization_result(str(BAYES_RESULTS_PICKLE))
    
    # Evaluate test performance
    best_param_values = [best_params[name] for name in optimizer.param_names]
    test_metrics = optimizer.evaluate_test_performance(best_param_values)
    print("\nTest Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Print best parameters
    print("\nBest Parameters:")
    for param_name, value in best_params.items():
        print(f"  {param_name}: {value:.6f}")

if __name__ == "__main__":
    main() 
