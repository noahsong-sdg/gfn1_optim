"""Base optimizer class for TBLite parameter optimization."""

import numpy as np
import pandas as pd
import toml
import tempfile
import os
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
import random

from calculators.calc import GeneralCalculator, DissociationCurveGenerator, CrystalGenerator, CalcConfig, CalcMethod
from calculators.tblite_ase_calculator import TBLiteASECalculator
from utils.data_extraction import extract_system_parameters
from config import get_system_config, CalculationType
from utils.parameter_bounds import ParameterBoundsManager, ParameterBounds
from common import setup_logging, RESULTS_DIR

logger = setup_logging(module_name="base_optimizer")

@dataclass
class BaseConfig:
    convergence_threshold: float = 1e-6
    patience: int = 20
    max_workers: int = 4


class BaseOptimizer(ABC):
    def __init__(self, system_name: str, base_param_file: str, 
                 reference_data: Optional[pd.DataFrame] = None,
                 train_fraction: float = 0.8, spin: int = 0,
                 method_name: Optional[str] = None):
        
        self.system_name = system_name
        self.system_config = get_system_config(system_name)
        
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        
        self.base_param_file = Path(base_param_file)
        self.train_fraction = train_fraction
        self.spin = spin
        
        self.bounds_manager = ParameterBoundsManager()
        self.parameter_bounds = self._define_parameter_bounds()
        
        self.full_reference_data = reference_data or self._load_or_generate_reference_data()
        self._split_train_test_data()
        
        # Optimization state
        self.best_parameters = None
        self.best_fitness = float('inf')
        self.convergence_counter = 0
        self.fitness_history = []
        self.failed_evaluations = 0
        
        self.method_name = method_name or self.__class__.__name__.replace('GeneralParameter', '').replace('BaseOptimizer', '').lower()
        
        # Load checkpoint if exists
        checkpoint_path = self.get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_checkpoint()
        
    def _define_parameter_bounds(self) -> List[ParameterBounds]:
        system_defaults = extract_system_parameters(self.system_config.elements)
        bounds = []
        
        for param_name, default_val in system_defaults.items():
            try:
                bound = self.bounds_manager.create_parameter_bounds(param_name, default_val)
                bounds.append(bound)
            except ValueError:
                continue
        
        logger.info(f"Generated {len(bounds)} parameter bounds for {self.system_name}")
        return bounds
    
    def apply_bounds(self, parameters: Dict[str, float]) -> Dict[str, float]:
        return self.bounds_manager.apply_bounds(parameters, self.parameter_bounds)
    
    def _load_or_generate_reference_data(self) -> pd.DataFrame:
        # Try CCSD data first
        if self.system_name in ["H2", "Si2"]:
            ccsd_file = RESULTS_DIR / "curves" / f"{self.system_name.lower()}_ccsd_500.csv"
            if ccsd_file.exists():
                return pd.read_csv(ccsd_file)
        
        # Try system-specific reference file
        ref_file = Path(self.system_config.reference_data_file)
        if ref_file.exists():
            return pd.read_csv(ref_file)
        
        # Generate fallback data
        logger.warning(f"Generating fallback data for {self.system_name}")
        calc_config = CalcConfig(method=CalcMethod.GFN1_XTB)
        calculator = GeneralCalculator(calc_config, self.system_config)
        
        if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
            generator = CrystalGenerator(calculator)
            return generator.generate_lattice_scan(save=True, filename=str(ref_file))
        else:
            generator = DissociationCurveGenerator(calculator)
            return generator.generate_curve(save=True, filename=str(ref_file))
    
    def _split_train_test_data(self):
        if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
            # Dummy values for solids
            self.train_distances = self.test_distances = np.array([])
            self.train_energies = self.test_energies = np.array([])
            self.reference_data = self.test_reference_data = pd.DataFrame()
            return
        
        # Split molecular data
        if 'Distance' in self.full_reference_data.columns:
            full_distances = self.full_reference_data['Distance'].values
            full_energies = self.full_reference_data['Energy'].values
        else:
            full_distances = self.full_reference_data.iloc[:, 0].values
            full_energies = self.full_reference_data.iloc[:, 1].values
        
        n_total = len(full_distances)
        n_train = int(n_total * self.train_fraction)
        
        np.random.seed(42)
        indices = np.random.permutation(n_total)
        train_indices, test_indices = indices[:n_train], indices[n_train:]
        
        self.train_distances = full_distances[train_indices]
        self.train_energies = full_energies[train_indices]
        self.test_distances = full_distances[test_indices]
        self.test_energies = full_energies[test_indices]
        
        self.reference_data = pd.DataFrame({'Distance': self.train_distances, 'Energy': self.train_energies})
        self.test_reference_data = pd.DataFrame({'Distance': self.test_distances, 'Energy': self.test_energies})
    
    def _set_parameter_in_dict(self, param_dict: dict, path: str, value: float):
        import re
        
        if hasattr(value, 'item'):
            value = value.item()
        elif isinstance(value, (np.floating, np.integer)):
            value = float(value) if isinstance(value, np.floating) else int(value)
        
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
    
    def create_param_file(self, parameters: Dict[str, float]) -> str:
        params = copy.deepcopy(self.base_params)
        for param_name, value in parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(params, f)
            return f.name
    

    
    def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
        if self.failed_evaluations > 10:
            raise ValueError("Too many failed evaluations")
        
        try:
            parameters = self.apply_bounds(parameters)
            param_file = self.create_param_file(parameters)

            if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
                calc_config = CalcConfig(method=CalcMethod.XTB_CUSTOM, param_file=param_file, spin=self.spin)
                calculator = GeneralCalculator(calc_config, self.system_config)
                generator = CrystalGenerator(calculator)
                result_df = generator.compute_stuff()
                a_opt, c_opt = result_df['a'].iloc[0], result_df['c'].iloc[0]
                a_ref, c_ref = self.system_config.lattice_params["a"], self.system_config.lattice_params["c"]
                loss = (a_opt - a_ref) ** 2 + (c_opt - c_ref) ** 2
                os.unlink(param_file)
                return loss

            # Molecular fitting
            calc_config = CalcConfig(method=CalcMethod.XTB_CUSTOM, param_file=param_file, spin=self.spin)
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            calc_data = generator.generate_curve(self.train_distances)
            os.unlink(param_file)
            
            # DEBUG: Print calc_data and reference_data for inspection before converting to float arrays
            print("[DEBUG] reference_data['Energy']:", self.reference_data['Energy'])
            print("[DEBUG] calc_data['Energy']:", calc_data['Energy'])

            ref_energies = self.reference_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            if len(ref_energies) != len(calc_energies):
                raise ValueError(f"Shape mismatch: {len(ref_energies)} vs {len(calc_energies)}")
            
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            return np.sqrt(np.mean((ref_relative - calc_relative)**2))
            
        except Exception as e:
            # DEBUG: Log the exception for debugging failed evaluations
            import traceback
            print(f"[DEBUG] Exception in evaluate_fitness: {e}")
            traceback.print_exc()
            if hasattr(self, 'logger'):
                self.logger.error(f"[DEBUG] Exception in evaluate_fitness: {e}")
            self.failed_evaluations += 1
            print(f"[DEBUG] Parameters for failed evaluation: {parameters}")
            return 1000.0
    
    def evaluate_test_performance(self, parameters: Dict[str, float]) -> Dict[str, float]:
        try:
            bounded_params = self.apply_bounds(parameters)
            param_file = self.create_param_file(bounded_params)

            if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
                calc_config = CalcConfig(method=CalcMethod.XTB_CUSTOM, param_file=param_file, spin=self.spin)
                calculator = GeneralCalculator(calc_config, self.system_config)
                generator = CrystalGenerator(calculator)
                result_df = generator.compute_stuff()
                a_opt, c_opt = result_df['a'].iloc[0], result_df['c'].iloc[0]
                a_ref, c_ref = self.system_config.lattice_params["a"], self.system_config.lattice_params["c"]
                a_error, c_error = abs(a_opt - a_ref), abs(c_opt - c_ref)
                os.unlink(param_file)
                return {'test_a_error': a_error, 'test_c_error': c_error, 'test_total_error': a_error + c_error}

            # Molecular test evaluation
            calc_config = CalcConfig(method=CalcMethod.XTB_CUSTOM, param_file=param_file, spin=self.spin)
            calculator = GeneralCalculator(calc_config, self.system_config)
            generator = DissociationCurveGenerator(calculator)
            
            calc_data = generator.generate_curve(self.test_reference_data['Distance'].values)
            os.unlink(param_file)
            
            ref_energies = self.test_reference_data['Energy'].values
            calc_energies = calc_data['Energy'].values
            
            ref_relative = ref_energies - np.min(ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            mae = np.mean(np.abs(ref_relative - calc_relative))
            max_error = np.max(np.abs(ref_relative - calc_relative))
            
            return {'test_rmse': rmse, 'test_mae': mae, 'test_max_error': max_error}
            
        except Exception as e:
            return {'test_rmse': 1000.0, 'test_mae': 1000.0, 'test_max_error': 1000.0}
    
    def check_convergence(self) -> bool:
        if len(self.fitness_history) < 2:
            return False
        
        threshold = getattr(self.config, 'convergence_threshold', 1e-6)
        patience = getattr(self.config, 'patience', 20)
        
        recent_improvement = abs(self.fitness_history[-2] - self.fitness_history[-1])
        
        if recent_improvement < threshold:
            self.convergence_counter += 1
            return self.convergence_counter >= patience
        else:
            self.convergence_counter = 0
            return False
    
    def get_method_specific_filename(self, base_filename: str) -> str:
        path = Path(base_filename)
        new_stem = f"{path.stem}_{self.method_name}"
        return str(path.parent / f"{new_stem}{path.suffix}")
    
    def get_optimized_params_filename(self) -> str:
        return self.get_method_specific_filename(self.system_config.optimized_params_file)
    
    def get_fitness_history_filename(self) -> str:
        return self.get_method_specific_filename(self.system_config.fitness_history_file)
    
    def save_best_parameters(self, filename: Optional[str] = None):
        if self.best_parameters is None:
            raise ValueError("No optimization has been run")
        
        filename = filename or self.get_optimized_params_filename()
        params = copy.deepcopy(self.base_params)
        for param_name, value in self.best_parameters.items():
            self._set_parameter_in_dict(params, param_name, value)
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            toml.dump(params, f)
        
        logger.info(f"Best parameters saved to {filename}")
    
    def save_fitness_history(self, filename: Optional[str] = None):
        if not self.fitness_history:
            raise ValueError("No optimization has been run")
        
        filename = filename or self.get_fitness_history_filename()
        
        if isinstance(self.fitness_history[0], dict):
            df = pd.DataFrame(self.fitness_history)
        else:
            df = pd.DataFrame({
                'iteration': range(len(self.fitness_history)),
                'best_fitness': self.fitness_history
            })
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Fitness history saved to {filename}")
    
    def get_checkpoint_path(self) -> str:
        return f"{self.system_name.lower()}_{self.method_name.lower()}_checkpt.pkl"

    def save_checkpoint(self):
        try:
            state = self.get_state()
            state['random_state'] = random.getstate()
            state['np_random_state'] = np.random.get_state()
            with open(self.get_checkpoint_path(), 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        try:
            with open(self.get_checkpoint_path(), 'rb') as f:
                state = pickle.load(f)
            self.set_state(state)
            if 'random_state' in state:
                random.setstate(state['random_state'])
            if 'np_random_state' in state:
                np.random.set_state(state['np_random_state'])
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            self.best_parameters = None
            self.best_fitness = float('inf')
            self.convergence_counter = 0
            self.fitness_history = []
            self.failed_evaluations = 0

    @abstractmethod
    def optimize(self) -> Dict[str, float]:
        pass
    
    def get_state(self) -> dict:
        return {
            'best_parameters': self.best_parameters,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'convergence_counter': getattr(self, 'convergence_counter', 0)
        }

    def set_state(self, state: dict):
        self.best_parameters = state.get('best_parameters')
        self.best_fitness = state.get('best_fitness', float('inf'))
        self.fitness_history = state.get('fitness_history', [])
        self.convergence_counter = state.get('convergence_counter', 0)
    
    def get_parameter_names(self) -> List[str]:
        return [bound.name for bound in self.parameter_bounds]
    
    def get_parameter_defaults(self) -> Dict[str, float]:
        return {bound.name: bound.default_val for bound in self.parameter_bounds}
    
    def get_parameter_bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        return {bound.name: (bound.min_val, bound.max_val) for bound in self.parameter_bounds} 
