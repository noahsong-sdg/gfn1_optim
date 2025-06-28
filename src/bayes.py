"""
Bayesian Optimization for TBLite parameter optimization
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
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
SKLEARN_AVAILABLE = True

from calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from config import get_system_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

# Reference data files  
CCSD_REFERENCE_DATA = RESULTS_DIR / "curves" / "h2_ccsd_data.csv"

# Output files
BAYESIAN_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "bayesian_optimized_params.toml"
BAYESIAN_FITNESS_HISTORY = RESULTS_DIR / "fitness" / "bayesian_fitness_history.csv"
BAYESIAN_CURVE_DATA = RESULTS_DIR / "curves" / "h2_bayesian_data.csv"

@dataclass
class BayesianConfig:
    """Configuration for Bayesian Optimization"""
    n_initial_points: int = 10
    n_iterations: int = 50
    acquisition_function: str = "ei"  # "ei", "pi", "ucb"
    xi: float = 0.01  # exploration parameter for EI and PI
    kappa: float = 2.576  # exploration parameter for UCB
    gp_kernel: str = "matern"  # "matern", "rbf"
    random_state: int = 42
    convergence_threshold: float = 1e-6
    patience: int = 10
    max_workers: int = 4

@dataclass
class ParameterBounds:
    """Parameter bounds for Bayesian optimization"""
    name: str
    min_val: float
    max_val: float
    default_val: float

class TBLiteParameterBayesian:
    """Bayesian Optimizer for TBLite parameters using H2 dissociation data"""
    
    def __init__(self, 
                 base_param_file: str,
                 reference_data: Optional[pd.DataFrame] = None,
                 config: BayesianConfig = BayesianConfig(),
                 train_fraction: float = 0.8):
        """Initialize Bayesian optimizer"""
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Bayesian optimization. "
                            "Install with: pip install scikit-learn")
        
        # Load base parameters
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        
        self.config = config
        self.train_fraction = train_fraction
        
        # System configuration for H2
        self.system_config = get_system_config("H2")
        
        # Define H2-relevant parameter bounds
        self.parameter_bounds = self._define_h2_parameter_bounds()
        self.param_names = [bound.name for bound in self.parameter_bounds]
        
        # Load and split reference data
        if reference_data is not None:
            self.full_reference_data = reference_data
        else:
            self.full_reference_data = self._load_reference_data()
        
        self._split_train_test_data()
        
        # Initialize optimization state
        self.X_observed = []  # Parameter vectors
        self.y_observed = []  # Fitness values
        self.iteration = 0
        self.convergence_counter = 0
        self.fitness_history = []
        self.failed_evaluations = 0
        self.best_params = {}
        self.best_fitness = float('inf')
        
        # Set random seed
        np.random.seed(self.config.random_state)
        
        # Initialize Gaussian Process
        self._initialize_gp()
        
    def _define_h2_parameter_bounds(self) -> List[ParameterBounds]:
        """Define parameter bounds for H2-relevant parameters"""
        bounds = []
        
        # Hamiltonian parameters
        bounds.extend([
            ParameterBounds("hamiltonian.xtb.kpol", 1.0, 5.0, 2.85),
            ParameterBounds("hamiltonian.xtb.enscale", -0.02, 0.02, -0.007),
        ])
        
        # Shell parameters
        bounds.extend([
            ParameterBounds("hamiltonian.xtb.shell.ss", 1.0, 3.0, 1.85),
            ParameterBounds("hamiltonian.xtb.shell.pp", 1.5, 3.5, 2.25),
            ParameterBounds("hamiltonian.xtb.shell.sp", 1.5, 3.0, 2.08),
        ])
        
        # H-H pair interaction
        bounds.append(ParameterBounds("hamiltonian.xtb.kpair.H-H", 0.5, 1.5, 0.96))
        
        # Hydrogen element parameters
        h_element_bounds = [
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
        
        for param_path, min_val, max_val, default in h_element_bounds:
            bounds.append(ParameterBounds(param_path, min_val, max_val, default))
            
        return bounds
    
    def _initialize_gp(self):
        """Initialize Gaussian Process model"""
        if self.config.gp_kernel == "matern":
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        elif self.config.gp_kernel == "rbf":
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        else:
            raise ValueError(f"Unknown kernel: {self.config.gp_kernel}")
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            random_state=self.config.random_state
        )
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load or generate reference CCSD data for H2"""
        if CCSD_REFERENCE_DATA.exists():
            logger.info(f"Loading reference data from {CCSD_REFERENCE_DATA}")
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
        
        logger.info(f"Training points: {len(self.train_distances)}, Testing points: {len(self.test_distances)}")
    
    def _set_parameter_in_dict(self, param_dict: dict, path: str, value: float):
        """Set a parameter value using dot notation path"""
        import re
        
        # Handle array access like 'element.H.levels[0]'
        if '[' in path and ']' in path:
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
        """Create a parameter file with the given parameters"""
        param_dict = self.base_params.copy()
        
        for param_name, value in parameters.items():
            self._set_parameter_in_dict(param_dict, param_name, value)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(param_dict, f)
            return f.name

    def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
        """Evaluate fitness of parameter set using H2 RMSE"""
        try:
            param_file = self.create_param_file(parameters)
            
            try:
                # Create calculator with custom parameters
                custom_config = CalcConfig(
                    method=CalcMethod.XTB_CUSTOM,
                    param_file=param_file,
                    spin=1
                )
                
                calculator = GeneralCalculator(custom_config, self.system_config)
                generator = DissociationCurveGenerator(calculator)
                
                # Calculate H2 curve on training distances
                calc_data = generator.generate_curve(
                    distances=self.train_distances, save=False
                )
                
                # Calculate RMSE vs reference
                ref_energies = self.train_data['Energy'].values
                calc_energies = calc_data['Energy'].values
                
                # Align to dissociation limit
                ref_relative = ref_energies - ref_energies[-1]
                calc_relative = calc_energies - calc_energies[-1]
                
                rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
                
                return rmse
                
            finally:
                # Clean up temporary file
                Path(param_file).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to evaluate fitness: {e}")
            self.failed_evaluations += 1
            return float('inf')

    def evaluate_test_performance(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Evaluate performance on test set"""
        try:
            param_file = self.create_param_file(parameters)
            
            try:
                custom_config = CalcConfig(
                    method=CalcMethod.XTB_CUSTOM,
                    param_file=param_file,
                    spin=1
                )
                
                calculator = GeneralCalculator(custom_config, self.system_config)
                generator = DissociationCurveGenerator(calculator)
                
                # Calculate on test distances
                calc_data = generator.generate_curve(
                    distances=self.test_distances, save=False
                )
                
                # Calculate metrics
                ref_energies = self.test_data['Energy'].values
                calc_energies = calc_data['Energy'].values
                
                ref_relative = ref_energies - ref_energies[-1]
                calc_relative = calc_energies - calc_energies[-1]
                
                rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
                mae = np.mean(np.abs(ref_relative - calc_relative))
                max_error = np.max(np.abs(ref_relative - calc_relative))
                
                return {
                    'rmse': rmse,
                    'mae': mae,
                    'max_error': max_error
                }
                
            finally:
                Path(param_file).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to evaluate test performance: {e}")
            return {'rmse': float('inf'), 'mae': float('inf'), 'max_error': float('inf')}

    def _normalize_parameters(self, parameters: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0,1] range for GP"""
        normalized = np.zeros(len(self.param_names))
        for i, param_name in enumerate(self.param_names):
            bound = next(b for b in self.parameter_bounds if b.name == param_name)
            normalized[i] = (parameters[param_name] - bound.min_val) / (bound.max_val - bound.min_val)
        return normalized

    def _denormalize_parameters(self, normalized: np.ndarray) -> Dict[str, float]:
        """Denormalize parameters from [0,1] range"""
        parameters = {}
        for i, param_name in enumerate(self.param_names):
            bound = next(b for b in self.parameter_bounds if b.name == param_name)
            parameters[param_name] = bound.min_val + normalized[i] * (bound.max_val - bound.min_val)
        return parameters

    def _generate_initial_points(self, n_points: int) -> List[Dict[str, float]]:
        """Generate initial points using Latin Hypercube sampling"""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=len(self.param_names), seed=self.config.random_state)
            samples = sampler.random(n=n_points)
        except ImportError:
            # Fallback to random sampling
            logger.warning("scipy.stats.qmc not available, using random sampling")
            samples = np.random.random((n_points, len(self.param_names)))
        
        points = []
        for sample in samples:
            parameters = self._denormalize_parameters(sample)
            points.append(parameters)
        
        return points

    def _acquisition_function(self, x: np.ndarray) -> float:
        """Compute acquisition function value"""
        if len(self.X_observed) == 0:
            return 0.0
            
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        if self.config.acquisition_function == "ei":
            # Expected Improvement
            if sigma == 0:
                return 0.0
            
            y_best = np.min(self.y_observed)
            z = (y_best - mu - self.config.xi) / sigma
            ei = (y_best - mu - self.config.xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei.item()  # Negative because we minimize
            
        elif self.config.acquisition_function == "pi":
            # Probability of Improvement
            if sigma == 0:
                return 0.0
                
            y_best = np.min(self.y_observed)
            z = (y_best - mu - self.config.xi) / sigma
            return -norm.cdf(z).item()  # Negative because we minimize
            
        elif self.config.acquisition_function == "ucb":
            # Upper Confidence Bound (for minimization, we use lower confidence bound)
            return (mu - self.config.kappa * sigma).item()
            
        else:
            raise ValueError(f"Unknown acquisition function: {self.config.acquisition_function}")

    def _optimize_acquisition(self) -> Dict[str, float]:
        """Optimize acquisition function to find next point"""
        best_value = float('inf')
        best_point = None
        
        # Multi-start optimization
        n_restarts = 10
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.random(len(self.param_names))
            
            bounds = [(0, 1) for _ in range(len(self.param_names))]
            
            result = minimize(
                self._acquisition_function,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_point = result.x
        
        if best_point is None:
            # Fallback to random point
            best_point = np.random.random(len(self.param_names))
        
        return self._denormalize_parameters(best_point)

    def optimize(self) -> Dict[str, float]:
        """Run Bayesian optimization"""
        logger.info(f"Starting Bayesian optimization with {self.config.n_iterations} iterations")
        start_time = time.time()
        
        # Phase 1: Initial exploration
        logger.info(f"Phase 1: Exploring with {self.config.n_initial_points} initial points")
        initial_points = self._generate_initial_points(self.config.n_initial_points)
        
        for params in initial_points:
            fitness = self.evaluate_fitness(params)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_params = params.copy()
            
            self.X_observed.append(self._normalize_parameters(params))
            self.y_observed.append(fitness)
            self.fitness_history.append(fitness)
            
            logger.info(f"Initial point {len(self.X_observed)}: fitness = {fitness:.6f}")
        
        # Phase 2: Bayesian optimization
        logger.info("Phase 2: Bayesian optimization")
        for iteration in range(self.config.n_iterations - self.config.n_initial_points):
            self.iteration = iteration + self.config.n_initial_points
            
            # Fit GP to observed data
            if len(self.X_observed) > 0:
                X = np.array(self.X_observed)
                y = np.array(self.y_observed)
                self.gp.fit(X, y)
            
            # Find next point using acquisition function
            next_params = self._optimize_acquisition()
            
            # Evaluate next point
            fitness = self.evaluate_fitness(next_params)
            
            # Update best
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_params = next_params.copy()
                self.convergence_counter = 0
            else:
                self.convergence_counter += 1
            
            # Record observation
            self.X_observed.append(self._normalize_parameters(next_params))
            self.y_observed.append(fitness)
            self.fitness_history.append(fitness)
            
            logger.info(f"Iteration {self.iteration + 1}: fitness = {fitness:.6f}, best = {self.best_fitness:.6f}")
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged after {self.iteration + 1} iterations")
                break
        
        elapsed = time.time() - start_time
        logger.info(f"Optimization completed in {elapsed:.2f} seconds")
        logger.info(f"Best fitness: {self.best_fitness:.6f}")
        logger.info(f"Failed evaluations: {self.failed_evaluations}")
        
        # Evaluate on test set
        test_performance = self.evaluate_test_performance(self.best_params)
        logger.info(f"Test performance - RMSE: {test_performance['rmse']:.6f}, "
                   f"MAE: {test_performance['mae']:.6f}, "
                   f"Max Error: {test_performance['max_error']:.6f}")
        
        return self.best_params

    def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if self.convergence_counter >= self.config.patience:
            return True
        
        if len(self.fitness_history) >= 10:
            recent_improvement = self.fitness_history[-10] - self.fitness_history[-1]
            if recent_improvement < self.config.convergence_threshold:
                return True
        
        return False

    def get_best_parameters(self) -> Dict[str, float]:
        """Get the best parameters found"""
        return self.best_params.copy()

    def save_best_parameters(self, filename: str):
        """Save best parameters to TOML file"""
        param_dict = self.base_params.copy()
        
        for param_name, value in self.best_params.items():
            self._set_parameter_in_dict(param_dict, param_name, value)
        
        with open(filename, 'w') as f:
            toml.dump(param_dict, f)
        
        logger.info(f"Best parameters saved to {filename}")

    def save_fitness_history(self, filename: str):
        """Save fitness history to CSV"""
        df = pd.DataFrame({
            'iteration': range(len(self.fitness_history)),
            'fitness': self.fitness_history
        })
        df.to_csv(filename, index=False)
        logger.info(f"Fitness history saved to {filename}")

def main():
    """Main function for testing"""
    config = BayesianConfig(
        n_initial_points=5,
        n_iterations=20,
        acquisition_function="ei"
    )
    
    optimizer = TBLiteParameterBayesian(
        base_param_file=str(BASE_PARAM_FILE),
        config=config
    )
    
    best_params = optimizer.optimize()
    
    # Save results
    optimizer.save_best_parameters(str(BAYESIAN_OPTIMIZED_PARAMS))
    optimizer.save_fitness_history(str(BAYESIAN_FITNESS_HISTORY))
    
    print("Bayesian optimization completed!")
    print(f"Best fitness: {optimizer.best_fitness:.6f}")

if __name__ == "__main__":
    main() 
