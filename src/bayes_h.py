"""
rn applied for h2
ctrl f for h2 and replace everything
and change the SPIN variable

need to go to main() to change the hyperparameters

"""

SPIN = 1


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
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_1cb
from skopt import dump, load 

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPT_DIR = PROJECT_ROOT / "src"

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
CCSD_REFERENCE_DATA = RESULTS_DIR / "curves" / "h2_ccsd_500.csv"
BAYES_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "bayes_optimized_params_v2.toml"
BAYES_FITNESS_HISTORY = RESULTS_DIR / "fitness" / "bayes_fitness_history_v2.csv"
BAYES_RESULTS_PICKLE = RESULTS_DIR / "parameters" / "bayes_results_v2.pkl"

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
    def __init__(self, base_param_file: str,
                 ref_data: Optional[pd.DataFrame] = None,
                 config: BayesConfig = BayesConfig(),
                 train_frac: float = 0.8):
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        self.config = config 
        self.train_frac = train_frac 
        self.parameter_space = self._define_h2_parameter_space()
        self.param_names = [param.name for param in self.parameter_space]
        self.dimensions = [
            Real(param.min_val, param.max_val, name=param.name, prior='uniform')
                 for param in self.parameter_space
        ]
        self.ref_data = ref_data 
        if self.ref_data is None:
            raise ValueError("Reference data is required for Bayesian optimization")
        
        self._SplitTrainTest()

        self.iteration = 0
        self.bestFitness = float('inf')
        self.bestParam = {}
        self.fitnessHistory = {}
        #self.convergence_counter = 0
        self.failed_evaluations = 0

    def _get_h2_bounds(self) -> List[ParamBounds]:
        # defined by 50% +/-
        space = []

        space.extend([
            ParamBounds("hamiltonian.xtb.kpol", 1.0, 5.0, 2.85),
            ParamBounds("hamiltonian.xtb.enscale", -0.02, 0.02, -0.007),
        ])

        space.extend([
            ParamBounds("hamiltonian.xtb.shell.ss", 1.0, 3.0, 1.85),
            ParamBounds("hamiltonian.xtb.shell.pp", 1.5, 3.5, 2.25),
            ParamBounds("hamiltonian.xtb.shell.sp", 1.5, 3.0, 2.08),
        ])

        h_element_space = [
            ParamBounds("element.H.levels[0]", -15.0, -8.0, -10.92),
            ParamBounds("element.H.levels[1]", -4.0, -1.0, -2.17),
            ParamBounds("element.H.slater[0]", 0.8, 2.0, 1.21),
            ParamBounds("element.H.slater[1]", 1.0, 3.0, 1.99),
            ParamBounds("element.H.kcn[0]", 0.01, 0.15, 0.0655),
            ParamBounds("element.H.kcn[1]", 0.001, 0.05, 0.0130),
            ParamBounds("element.H.gam", 0.2, 0.8, 0.47),
        ]

        space.extend(h_element_space)
        return space 
    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(CCSD_REFERENCE_DATA)
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
        from config import get_system_config
        custom_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=param_file,
            spin=SPIN
        )
        system_config = get_system_config("H2")
        calculator = GeneralCalculator(custom_config, system_config)
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
            self.convergence_counter = 0
        else:
            self.convergence_counter += 1
        self.fitnessHistory.append(rmse)
        Path(param_file).unlink(missing_ok=True)
        return rmse 
    
    def _evaluate_test_performance(self, paramVals: List[float]) -> Dict[str, float]:
        param_file = self._create_param_file(paramVals)
        from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
        from config import get_system_config
        custom_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=param_file,
            spin=SPIN
        )

        system_config = get_system_config("H2")
        calculator = GeneralCalculator(custom_config, system_config)
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
        acq_fn = self.config.acquisition_func.upper()
        if acq_fn == "EI":
            acq_fn = gaussian_ei
        elif acq_fn == "PI":
            acq_fn = gaussian_pi
        elif acq_fn == "LCB":
            acq_fn = gaussian_1cb
        else:
            raise ValueError(f"Unknown acquisition function: {acq_fn}")
        
        x0 = []
        y0 = []
        
        default_values = [param.default_val for param in self.parameter_space]
        print("Evaluating default parameters...")
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
            acq_func=acq_fn,
            acq_optimizer="auto",
            n_restarts_optimizer=self.config.n_restarts,
            noise=self.config.noise,
            random_state=self.config.random_state,
            # callback=self._callback,
            model_queue_size=1
        )

        self.best_params = dict(zip(self.param_names, result.x))
        self.best_fitness = result.fun
        self.optimization_result = result
        
        return self.best_params

    def get_best_params(self) -> Dict[str, float]:
         if not self.best_params:
            raise ValueError("No optimization has been run")
         return self.best_params.copy()
    
    def save_best_params(self, filename: str):
        if not self.best_params:
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
        if not self.optimization_result:
            raise ValueError("No optimization has been run")
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        dump(self.optimization_result, filename)

if __name__ == "__main__":
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
        base_param_file=str(BASE_PARAM_FILE),
        config=bayes_config
    )
    best_params = optim.optimize()
    print("Best parameters:", best_params)
    optim.save_best_params(BAYES_OPTIMIZED_PARAMS)
    optim.save_fitness_history(BAYES_FITNESS_HISTORY)
    optim.save_optimization_result(BAYES_RESULTS_PICKLE)

    # Evaluate test performance
    best_param_values = [best_params[name] for name in optim.param_names]
    test_metrics = optim.evaluate_test_performance(best_param_values)
    print("\nTest Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
