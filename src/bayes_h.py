"""
rn applied for h2
ctrl f for h2 and replace everything
and change the SPIN variable

need to go to main() to change the hyperparameters

"""

SPIN = 2


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import toml
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple 
from dataclasses import dataclass 
import time

from sklearn.model_selection import train_test_split
from skopt import gp_minimize 
from skopt.space import Real  
from skopt import dump, load 

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPT_DIR = PROJECT_ROOT / "src"

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
CCSD_REFERENCE_DATA = RESULTS_DIR / "curves" / "si2_ccsd_500.csv"
# these dont do anything rn 
BAYES_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "si_optim.toml"
BAYES_FITNESS_HISTORY = RESULTS_DIR / "fitness" / "si_fitness_history.csv"
BAYES_RESULTS_PICKLE = RESULTS_DIR / "parameters" / "si_results.pkl"

@dataclass 
class BayesConfig:
    n_calls: int = 100
    n_init_pts: int = 20
    n_restarts: int = 10
    acq_fn: str = "EI"
    acq_weight: float = 0.01
    random_state: int = 42
    convergence_threshold: float = 1e-6
    patience: int = 15
    noise: float = 1e-10

@dataclass
class ParamBounds:
    name: str 
    min_val: float 
    max_val: float 
    default_val: float 

class TBLiteBayesian:
    def __init__(self, 
                 system_name: str,
                 base_param_file: str,
                 ref_data: Optional[pd.DataFrame] = None,
                 config: BayesConfig = BayesConfig(),
                 train_frac: float = 0.8):
        """Initialize Bayesian optimizer
        
        Args:
            system_name: Name of system to optimize (e.g., 'H2', 'Si2', 'C2')
            base_param_file: Path to base parameter TOML file
            ref_data: Optional reference data (if None, loads from system config)
            config: Bayesian optimization configuration
            train_frac: Fraction of data to use for training
        """
        
        # System configuration
        self.system_name = system_name
        from config import get_system_config
        self.system_config = get_system_config(system_name)
        
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        self.config = config 
        self.train_frac = train_frac 
        
        # Define parameter space using automatic extraction with 50% +/- bounds
        self.parameter_space = self._define_parameter_space()
        self.param_names = [param.name for param in self.parameter_space]
        self.dimensions = [
            Real(param.min_val, param.max_val, name=param.name, prior='uniform')
                 for param in self.parameter_space
        ]
        
        # Load and split reference data
        self.ref_data = ref_data 
        if self.ref_data is None:
            self.ref_data = self._load_reference_data()
        
        if self.ref_data.empty:
            raise ValueError(f"No reference data available for {system_name}")
        
        self._SplitTrainTest()

        self.iteration = 0
        self.bestFitness = float('inf')
        self.bestParam = {}
        self.fitnessHistory = []
        self.failed_evaluations = 0

    def _define_parameter_space(self) -> List[ParamBounds]:
        """Define parameter space using automatic extraction with 50% +/- margins"""
        space = []
        
        # Extract default parameters for this system
        from data_extraction import GFN1ParameterExtractor
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
            
            space.append(ParamBounds(param_name, min_val, max_val, default_val))
        
        logger.info(f"Generated {len(space)} parameter bounds for {self.system_name} with 50% +/- margins")
        return space
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference data for the system"""
        ref_file = CCSD_REFERENCE_DATA
        if ref_file.exists():
            logger.info(f"Loading reference data from {ref_file}")
            return pd.read_csv(ref_file)
        else:
            logger.warning(f"Reference file {ref_file} not found for {self.system_name}.")
            return pd.DataFrame(columns=['Distance', 'Energy'])
    
    def _SplitTrainTest(self):
        n = int(len(self.ref_data) * self.train_frac)
        train_idx = np.linspace(0, len(self.ref_data) - 1, n, dtype=int)
        test_idx = [i for i in range(len(self.ref_data)) if i not in train_idx]
        self.train_data = self.ref_data.iloc[train_idx].copy()
        self.test_data = self.ref_data.iloc[test_idx].copy()
        self.train_distances = self.train_data['Distance'].values
        self.test_distances = self.test_data['Distance'].values

    def _set_param_in_dict(self, param_dict: dict, path: str, value: float):
        # this is some witchcraft
        import re 
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
        keys = path.split('.')
        current = param_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _create_param_file(self, paramVals: List[float]) -> str:
        params = self.base_params.copy()
        parameterDict = dict(zip(self.param_names, paramVals))
        for param_name, val in parameterDict.items():
            self._set_param_in_dict(params, param_name, val)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(params, f)
            return f.name
    
    def _objective_function(self, paramVals: List[float]) -> float:
        self.iteration += 1
        param_file = self._create_param_file(paramVals)
        from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
        
        custom_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=param_file,
            spin=SPIN
        )
        
        calculator = GeneralCalculator(custom_config, self.system_config)
        generator = DissociationCurveGenerator(calculator)
        calc_data = generator.generate_curve(distances=self.train_distances, save=False)

        ref_energies = self.train_data['Energy'].values
        calc_energies = calc_data['Energy'].values

        ref_relative = ref_energies - np.min(ref_energies)
        calc_relative = calc_energies - np.min(calc_energies)
        rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))

        if rmse < self.bestFitness:
            self.bestFitness = rmse
            self.bestParam = dict(zip(self.param_names, paramVals))
            print(f"Evaluation {self.iteration}: New best fitness: {rmse:.6f}")
        
        self.fitnessHistory.append(rmse)
        Path(param_file).unlink(missing_ok=True)
        return rmse 
    
    def _evaluate_test_performance(self, paramVals: List[float]) -> Dict[str, float]:
        param_file = self._create_param_file(paramVals)
        from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
        
        custom_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=param_file,
            spin=SPIN
        )

        calculator = GeneralCalculator(custom_config, self.system_config)
        generator = DissociationCurveGenerator(calculator)
        calc_data = generator.generate_curve(distances=self.test_distances, save=False)
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
    
    def optimize(self) -> Dict[str, float]:
        # Use string acquisition function names (modern scikit-optimize)
        acq_fn = self.config.acq_fn.upper()
        if acq_fn not in ["EI", "PI", "LCB", "MES", "PVRS"]:
            logger.warning(f"Unknown acquisition function: {acq_fn}. Using 'EI' as default.")
            acq_fn = "EI"
        
        x0 = []
        y0 = []
        
        default_values = [param.default_val for param in self.parameter_space]
        print(f"Evaluating default parameters for {self.system_name}...")
        default_fitness = self._objective_function(default_values)
        x0.append(default_values)
        y0.append(default_fitness)

        def objective_wrapper(params):
            return self._objective_function(params)
        
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.dimensions,
            n_calls=self.config.n_calls,
            n_initial_points=self.config.n_init_pts - 1,
            x0=x0,
            y0=y0,
            acq_func=acq_fn,  # Now using string instead of function object
            acq_optimizer="auto",
            n_restarts_optimizer=self.config.n_restarts,
            noise=self.config.noise,
            random_state=self.config.random_state,
            model_queue_size=1
        )

        self.best_params = dict(zip(self.param_names, result.x))
        self.best_fitness = result.fun
        self.optimization_result = result
        
        return self.best_params

    def get_best_params(self) -> Dict[str, float]:
         if not hasattr(self, 'best_params'):
            raise ValueError("No optimization has been run")
         return self.best_params.copy()
    
    def save_best_params(self, filename: str):
        if not hasattr(self, 'best_params'):
            raise ValueError("No optimization has been run")
        
        params = self.base_params.copy()
        for param_name, value in self.best_params.items():
            self._set_param_in_dict(params, param_name, value)
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            toml.dump(params, f)
    
    def save_fitness_history(self, filename: str):
        df = pd.DataFrame({
            'evaluation': range(1, len(self.fitnessHistory) + 1),
            'best_fitness': self.fitnessHistory
        })
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False)
    
    def save_optimization_result(self, filename: str):
        if not hasattr(self, 'optimization_result'):
            raise ValueError("No optimization has been run")
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        dump(self.optimization_result, filename)

def main():
    """System-agnostic Bayesian optimization"""
    import sys
    
    # Allow system selection from command line
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
    else:
        system_name = "H2"  # Default
    
    print(f"Running Bayesian optimization for {system_name}")
    
    bayes_config = BayesConfig(
        n_calls=100,
        n_init_pts=20,
        n_restarts=10,
        acq_fn="EI",
        acq_weight=0.01,
        random_state=42,
        convergence_threshold=1e-6,
        patience=15
    )
    
    optim = TBLiteBayesian(
        system_name=system_name,
        base_param_file=str(BASE_PARAM_FILE),
        config=bayes_config
    )
    
    best_params = optim.optimize()
    print(f"Best parameters for {system_name}:", best_params)
    
    # Save results with system-specific filenames
    bayes_params_file = RESULTS_DIR / "parameters" / f"bayes_optimized_params_{system_name.lower()}.toml"
    bayes_history_file = RESULTS_DIR / "fitness" / f"bayes_fitness_history_{system_name.lower()}.csv"
    bayes_results_file = RESULTS_DIR / "parameters" / f"bayes_results_{system_name.lower()}.pkl"
    
    optim.save_best_params(str(bayes_params_file))
    optim.save_fitness_history(str(bayes_history_file))
    optim.save_optimization_result(str(bayes_results_file))

    # Evaluate test performance
    best_param_values = [best_params[name] for name in optim.param_names]
    test_metrics = optim._evaluate_test_performance(best_param_values)
    print(f"\nTest Performance for {system_name}:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

if __name__ == "__main__":
    main()
