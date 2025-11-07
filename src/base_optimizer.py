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
import ase.io
from ase import Atoms

from calculators.calc import GeneralCalculator, DissociationCurveGenerator, CrystalGenerator, CalcConfig, CalcMethod
from calculators.dftbp import run_dftbp_bandgap
from utils.extract_default import extract_system_parameters
from config import get_system_config, CalculationType
from utils.parameter_bounds import ParameterBoundsManager, ParameterBounds, init_dynamic_bounds
from common import setup_logging, RESULTS_DIR, RANDOM_SEED

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
        
        # Core setup
        self.system_name = system_name
        self.system_config = get_system_config(system_name)
        self.base_param_file = Path(base_param_file)
        self.train_fraction = train_fraction
        self.spin = spin
        
        # Load base parameters
        with open(base_param_file, 'r') as f:
            self.base_params = toml.load(f)
        
        # Setup parameter bounds
        self.bounds_manager = ParameterBoundsManager()
        system_defaults = extract_system_parameters(self.system_config.elements)
        self.parameter_bounds = init_dynamic_bounds(system_defaults)
        
        # Setup reference data and training split
        self._setup_reference_data(reference_data)
        
        # Initialize optimization state
        self._init_optimization_state(method_name)
        
        # Load checkpoint if exists
        self._load_checkpoint_if_exists()

    
    def apply_bounds(self, parameters: Dict[str, float]) -> Dict[str, float]:
        return self.bounds_manager.apply_bounds(parameters, self.parameter_bounds)
    
    def _setup_reference_data(self, reference_data: Optional[pd.DataFrame]):
        """Setup reference data based on calculation type"""
        if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
            self._setup_lattice_reference()
        else:
            if reference_data is not None:
                self.full_reference_data = reference_data
            else:
                self.full_reference_data = self._load_reference_data()
        
        self._split_train_test_data()
    
    def _setup_lattice_reference(self):
        """Setup reference data from experimental lattice parameters"""
        logger.info(f"Using experimental lattice parameters for {self.system_name}")
        import pandas as pd
        
        self.full_reference_data = pd.DataFrame([{
            'a': self.system_config.lattice_params['a'],
            'b': self.system_config.lattice_params['b'], 
            'c': self.system_config.lattice_params['c'],
            'alpha': self.system_config.lattice_params['alpha'],
            'beta': self.system_config.lattice_params['beta'],
            'gamma': self.system_config.lattice_params['gamma'],
            'energy': self.system_config.lattice_params['energy']
        }])
        
        logger.info(f"Reference: a={self.system_config.lattice_params['a']:.3f} Å, c={self.system_config.lattice_params['c']:.3f} Å")
    
    def _init_optimization_state(self, method_name: Optional[str]):
        """Initialize optimization state variables"""
        self.best_parameters = None
        self.best_fitness = float('inf')
        self.convergence_counter = 0
        self.fitness_history = []
        self.failed_evaluations = 0
        self.success_evaluations = 0
        
        self.method_name = method_name or self.__class__.__name__.replace('GeneralParameter', '').replace('BaseOptimizer', '').lower()
    
    def _load_checkpoint_if_exists(self):
        """Load checkpoint if it exists"""
        checkpoint_path = self.get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_checkpoint()
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference data for dissociation curves and bulk materials"""
        # Handle bulk materials
        if self.system_config.calculation_type == CalculationType.BULK:
            # Read-only: expect external CSV at self.system_config.reference_data_file
            ref_file = Path(self.system_config.reference_data_file)
            if ref_file.exists():
                logger.info(f"Loading bulk references from {ref_file}")
                try:
                    return pd.read_csv(ref_file)
                except Exception as e:
                    logger.warning(f"Failed to read bulk reference CSV {ref_file}: {e}")
            # No in-memory reference table needed for BULK if per-structure info is attached later
            return pd.DataFrame()
        
        # Try CCSD data for known systems
        if self.system_name in ["H2", "Si2"]:
            ccsd_file = RESULTS_DIR / "curves" / f"{self.system_name.lower()}_ccsd_500.csv"
            if ccsd_file.exists():
                return pd.read_csv(ccsd_file)
        
        # Try system-specific reference file
        ref_file = Path(self.system_config.reference_data_file)
        if ref_file.exists():
            return pd.read_csv(ref_file)
        
        # Generate fallback data as last resort
        logger.warning(f"Generating fallback data for {self.system_name}")
        return pd.DataFrame()
    
    def _load_bulk_reference_data(self) -> pd.DataFrame:
        """Load reference data for bulk materials from XYZ files"""
        xyz_file = self._get_bulk_xyz_file()
        
        if not Path(xyz_file).exists():
            raise FileNotFoundError(f"Bulk materials XYZ file not found: {xyz_file}")
        
        # Process the bulk system directly
        reference_df, _ = self._process_bulk_system(xyz_file, max_structures=self.system_config.num_points)
        
        # Save reference data for future use
        ref_file = Path(self.system_config.reference_data_file)
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        reference_df.to_csv(ref_file, index=False)
        logger.info(f"Saved reference data to {ref_file}")
        
        return reference_df
    
    def _split_train_test_data(self):
        if self.system_config.calculation_type in [CalculationType.LATTICE_CONSTANTS, CalculationType.BULK]:
            # For solids and bulk materials, use all data (no train/test split needed for energy optimization)
            self.train_distances = self.test_distances = np.array([])
            self.train_energies = self.test_energies = np.array([])
            self.reference_data = self.full_reference_data
            self.test_reference_data = self.full_reference_data
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
        
        np.random.seed(RANDOM_SEED)
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
        try:
            parameters = self.apply_bounds(parameters)
            param_file = self.create_param_file(parameters)

            if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
                calc_config = CalcConfig(method=CalcMethod.XTB_CUSTOM, param_file=param_file, spin=self.spin)
                calculator = GeneralCalculator(calc_config, self.system_config)
                generator = CrystalGenerator(calculator)
                result_df = generator.compute_stuff()
                #a_opt, c_opt, gap = result_df['a'].iloc[0], result_df['c'].iloc[0], result_df['bandgap'].iloc[0]
                a_opt, b_opt, c_opt = result_df['a'].iloc[0], result_df['b'].iloc[0], result_df['c'].iloc[0]
                alpha_opt, beta_opt, gamma_opt = result_df['alpha'].iloc[0], result_df['beta'].iloc[0], result_df['gamma'].iloc[0]
                a_ref, b_ref, c_ref = self.system_config.lattice_params["a"], self.system_config.lattice_params["b"], self.system_config.lattice_params["c"]
                
                # Target angles for wurtzite structure
                alpha_ref, beta_ref, gamma_ref = self.system_config.lattice_params["alpha"], self.system_config.lattice_params["beta"], self.system_config.lattice_params["gamma"]
                
                # Combined loss: lattice parameters + angles + energy
                # Normalize each component to [0,1] range and apply weights
                
                # Lattice parameter errors (normalized by typical scale ~0.1 Å)
                lattice_scale = 0.1  # typical error scale in Å
                lattice_loss = ((a_opt - a_ref) ** 2 + (b_opt - b_ref) ** 2 + (c_opt - c_ref) ** 2) / (3 * lattice_scale ** 2)
                
                # Angle errors (normalized by typical scale ~1 degree)
                angle_scale = 1.0  # typical error scale in degrees
                angle_loss = ((alpha_opt - alpha_ref) ** 2 + (beta_opt - beta_ref) ** 2 + (gamma_opt - gamma_ref) ** 2) / (3 * angle_scale ** 2)
                
                # Energy error (normalized by typical scale ~0.1 Hartree)
                energy_scale = 0.1  # typical error scale in Hartree
                energy_loss = (result_df['energy'].iloc[0] - self.system_config.lattice_params["energy"]) ** 2 / (energy_scale ** 2)
                
                # Debug output to check all values
                logger.info(f"Lattice comparison: calc a={a_opt:.3f}, b={b_opt:.3f}, c={c_opt:.3f} Å vs ref a={a_ref:.3f}, b={b_ref:.3f}, c={c_ref:.3f} Å")
                logger.info(f"Energy comparison: calc={result_df['energy'].iloc[0]:.6f} Hartree, ref={self.system_config.lattice_params['energy']:.6f} Hartree, diff={abs(result_df['energy'].iloc[0] - self.system_config.lattice_params['energy']):.6f} Hartree")
                
                # Weighted combination (energy is most important)
                lattice_weight = 0.10   
                angle_weight = 0.05     
                energy_weight = 0.85        
                
                total_loss = (lattice_weight * lattice_loss + 
                             angle_weight * angle_loss + 
                             energy_weight * energy_loss)
                
                # Convert to fitness (higher is better)
                fitness = 1.0 / (1.0 + total_loss)
                
                os.unlink(param_file)
                self.success_evaluations += 1
                return fitness

            elif self.system_config.calculation_type == CalculationType.BULK:
                # For bulk/supercell workflows, compute energies and bandgaps for provided
                # structures and compare to reference properties if available.
                xyz_file = self._get_bulk_xyz_file()
                if not Path(xyz_file).exists():
                    os.unlink(param_file)
                    raise FileNotFoundError(f"Bulk materials XYZ file not found: {xyz_file}")
                structures = self._load_structures_from_xyz(xyz_file, max_structures=self.system_config.num_points)
                # Calculate energies and bandgaps with current parameters
                energies, bandgaps = self._calculate_energies_and_bandgaps(structures, param_file)
                os.unlink(param_file)
                self.success_evaluations += 1

                # Energy component
                ref_energies = self._extract_reference_energies(structures)
                valid_mask_e = [not np.isnan(e) for e in energies]
                calc_e = np.array([e for e, ok in zip(energies, valid_mask_e) if ok], dtype=float)
                energy_rmse = None
                if any(not np.isnan(x) for x in ref_energies):
                    ref_e = np.array([r for r, ok in zip(ref_energies, valid_mask_e) if ok], dtype=float)
                    # Use relative energies to be robust to absolute offsets
                    ref_rel = ref_e - np.nanmin(ref_e)
                    calc_rel = calc_e - np.nanmin(calc_e)
                    energy_rmse = float(np.sqrt(np.mean((ref_rel - calc_rel) ** 2)))
                else:
                    # Without references, prefer lower energies across the dataset (use mean energy as proxy)
                    mean_e = float(np.nanmean(calc_e))
                    # Treat mean energy deviation as loss on a scale of its magnitude
                    energy_rmse = abs(mean_e)

                # Bandgap component
                ref_bandgaps = self._extract_reference_bandgaps(structures)
                valid_mask_g = [not np.isnan(g) for g in bandgaps]
                calc_g = np.array([g for g, ok in zip(bandgaps, valid_mask_g) if ok], dtype=float)
                gap_rmse = None
                if any(not np.isnan(x) for x in ref_bandgaps) and len(calc_g) > 0:
                    ref_g = np.array([r for r, ok in zip(ref_bandgaps, valid_mask_g) if ok], dtype=float)
                    gap_rmse = float(np.sqrt(np.mean((ref_g - calc_g) ** 2)))

                # Combine losses (normalize with simple scales; energy relative nature already reduces scale issues)
                energy_weight = 0.2
                gap_weight = 0.8 if gap_rmse is not None else 0.0
                total_loss = energy_weight * energy_rmse + gap_weight * (gap_rmse or 0.0)
                fitness = 1.0 / (1.0 + total_loss)

                return fitness


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
            
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            
            # Convert RMSE to fitness (higher is better)
            fitness = 1.0 / (1.0 + rmse)
            
            return fitness
            
        except Exception as e:
            # DEBUG: Log the exception for debugging failed evaluations
            import traceback
            print(f"[DEBUG] Exception in evaluate_fitness: {e}")
            traceback.print_exc()
            if hasattr(self, 'logger'):
                self.logger.error(f"[DEBUG] Exception in evaluate_fitness: {e}")
            self.failed_evaluations += 1
            print(f"[DEBUG] Parameters for failed evaluation: {parameters}")
            return 0.0  # Return very low fitness for failed evaluations
    
    def evaluate_test_performance(self, parameters: Dict[str, float]) -> Dict[str, float]:
        try:
            bounded_params = self.apply_bounds(parameters)
            param_file = self.create_param_file(bounded_params)

            if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
                calc_config = CalcConfig(method=CalcMethod.XTB_CUSTOM, param_file=param_file, spin=self.spin)
                calculator = GeneralCalculator(calc_config, self.system_config)
                generator = CrystalGenerator(calculator)
                result_df = generator.compute_stuff()
                a_opt, b_opt, c_opt = result_df['a'].iloc[0], result_df['b'].iloc[0], result_df['c'].iloc[0]
                a_ref, b_ref, c_ref = self.system_config.lattice_params["a"], self.system_config.lattice_params["b"], self.system_config.lattice_params["c"]
                a_error, b_error, c_error = abs(a_opt - a_ref), abs(b_opt - b_ref), abs(c_opt - c_ref)
                os.unlink(param_file)
                return {'test_a_error': a_error, 'test_b_error': b_error, 'test_c_error': c_error, 'test_total_error': a_error + b_error + c_error}

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
    
    def log_best_rmse(self):
        """Log the best fitness achieved during optimization"""
        if self.best_fitness is not None and self.best_fitness != float('inf'):
            logger.info(f"Best fitness: {self.best_fitness:.6f}")
        else:
            logger.info("Best fitness: N/A (no valid optimization completed)")
    
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
        self.log_best_rmse()  # Log the best RMSE
    
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
    
    # Bulk materials calculation methods
    def _get_bulk_xyz_file(self) -> str:
        """Get the appropriate XYZ file for bulk materials"""
        # Default dataset for bulk/supercell training
        return "trainall.xyz"
    
    def _load_structures_from_xyz(self, xyz_file: str, max_structures: Optional[int] = None) -> List[Atoms]:
        """Load structures from XYZ file"""
        logger.info(f"Loading structures from {xyz_file}")
        structures = list(ase.io.read(xyz_file, index=':'))
        
        if max_structures:
            structures = structures[:max_structures]
            
        logger.info(f"Loaded {len(structures)} structures")
        # Attach reference properties if available
        try:
            self._attach_bulk_references(structures)
        except Exception as e:
            logger.warning(f"Failed to attach bulk references: {e}")
        return structures
    
    def _extract_reference_energies(self, structures: List[Atoms]) -> List[float]:
        """Extract reference energies (eV) from structure properties."""
        energies = []
        for i, atoms in enumerate(structures):
            energy = np.nan
            if hasattr(atoms, 'info'):
                info = atoms.info
                if 'ref_energy_eV' in info:
                    try:
                        energy = float(info['ref_energy_eV'])
                    except Exception:
                        energy = np.nan
            energies.append(energy)
        logger.info(f"Collected reference energies for {len(energies)} structures")
        return energies
    
    def _calculate_energies(self, structures: List[Atoms], param_file: str) -> List[float]:
        """Calculate energies for all structures using DFTB+ runner, return in eV."""
        energies = []
        failed = 0
        for i, atoms in enumerate(structures):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Write structure file for dftbp
                    struct_path = Path(tmpdir) / "input.xyz"
                    ase.io.write(str(struct_path), atoms)
                    # Choose k-points based on periodicity
                    kpts = (1, 1, 1)
                    bandgap, energy = run_dftbp_bandgap(
                        str(struct_path),
                        kpts=kpts,
                        method="GFN1-xTB",
                        temp=400.0,
                        parameter_file=param_file,
                        workdir=tmpdir,
                    )
                    energies.append(float(energy) if energy is not None else np.nan)
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(structures)} structures")
            except Exception as e:
                logger.warning(f"Failed to calculate energy for structure {i}: {e}")
                failed += 1
                energies.append(np.nan)
        logger.info(f"Calculated energies for {len(energies) - failed}/{len(structures)} structures")
        return energies

    def _calculate_energies_and_bandgaps(self, structures: List[Atoms], param_file: str) -> Tuple[List[float], List[float]]:
        """Calculate energies and bandgaps for all structures using DFTB+ runner.
        Bandgaps may be NaN if not available/parsed.
        """
        energies = []
        bandgaps = []
        failed = 0
        for i, atoms in enumerate(structures):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    struct_path = Path(tmpdir) / "input.xyz"
                    ase.io.write(str(struct_path), atoms)
                    kpts = (1, 1, 1)
                    bg, e = run_dftbp_bandgap(
                        str(struct_path),
                        kpts=kpts,
                        method="GFN1-xTB",
                        temp=400.0,
                        parameter_file=param_file,
                        workdir=tmpdir,
                    )
                    energies.append(float(e) if e is not None else np.nan)
                    bandgaps.append(float(bg) if bg is not None else np.nan)
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(structures)} structures")
            except Exception as e:
                logger.warning(f"Failed to calculate properties for structure {i}: {e}")
                failed += 1
                energies.append(np.nan)
                bandgaps.append(np.nan)
        logger.info(f"Calculated energies for {len(energies) - failed}/{len(structures)} structures")
        return energies, bandgaps

    def _extract_reference_bandgaps(self, structures: List[Atoms]) -> List[float]:
        """Extract reference bandgaps from structure properties if present (eV)."""
        gaps = []
        for i, atoms in enumerate(structures):
            g = np.nan
            if hasattr(atoms, 'info'):
                info = atoms.info
                if 'ref_gap_eV' in info:
                    try:
                        g = float(info['ref_gap_eV'])
                    except Exception:
                        g = np.nan
            gaps.append(g)
        return gaps

    def _attach_bulk_references(self, structures: List[Atoms]) -> None:
        """Attach reference energy/gap (eV) from CSV to each structure.info.
        Expects CSV at self.system_config.reference_data_file with columns:
        'Structure', 'Bandgap(eV)', 'FreeEnergy(eV)'.
        If counts match, rows are aligned to structure order.
        """
        ref_path = Path(self.system_config.reference_data_file)
        if not ref_path.exists():
            logger.warning(f"Reference CSV not found at {ref_path}")
            return
        df = pd.read_csv(ref_path)
        n_csv = len(df)
        n_struct = len(structures)
        if n_csv != n_struct:
            logger.warning(f"CSV entries ({n_csv}) != structures ({n_struct}); aligning by min length")
        n = min(n_csv, n_struct)
        for i in range(n):
            row = df.iloc[i]
            atoms = structures[i]
            if not hasattr(atoms, 'info'):
                continue
            atoms.info['ref_structure'] = str(row.get('Structure', f'structure_{i:03d}'))
            if 'FreeEnergy(eV)' in row:
                try:
                    atoms.info['ref_energy_eV'] = float(row['FreeEnergy(eV)'])
                except Exception:
                    atoms.info['ref_energy_eV'] = np.nan
            if 'Bandgap(eV)' in row:
                try:
                    atoms.info['ref_gap_eV'] = float(row['Bandgap(eV)'])
                except Exception:
                    atoms.info['ref_gap_eV'] = np.nan
    
    def _compute_energy_loss(self, reference_energies: List[float], calculated_energies: List[float]) -> float:
        """Compute RMSE loss between reference and calculated energies"""
        valid_pairs = [(ref, calc) for ref, calc in zip(reference_energies, calculated_energies) 
                      if np.isfinite(ref) and np.isfinite(calc)]
        if not valid_pairs:
            n_ref = sum(np.isfinite(r) for r in reference_energies)
            n_calc = sum(np.isfinite(c) for c in calculated_energies)
            logger.warning(f"No valid energy pairs found (ref finite={n_ref}, calc finite={n_calc}); skipping RMSE computation")
            return float('nan')
        ref_vals, calc_vals = zip(*valid_pairs)
        ref_vals = np.array(ref_vals)
        calc_vals = np.array(calc_vals)
        rmse = np.sqrt(np.mean((ref_vals - calc_vals) ** 2))
        logger.info(f"Energy RMSE: {rmse:.6f} eV ({len(valid_pairs)} structures)")
        return rmse
    
    def _process_bulk_system(self, xyz_file: str, max_structures: Optional[int] = None) -> Tuple[pd.DataFrame, float]:
        """Process bulk system: load structures, calculate energies, compute loss"""
        structures = self._load_structures_from_xyz(xyz_file, max_structures)
        reference_energies = self._extract_reference_energies(structures)
        calculated_energies = self._calculate_energies(structures, self.base_param_file)
        rmse = self._compute_energy_loss(reference_energies, calculated_energies)
        
        # Create reference DataFrame
        data = []
        for i, (atoms, energy) in enumerate(zip(structures, reference_energies)):
            if np.isnan(energy):
                continue
                
            formula = atoms.get_chemical_formula()
            n_atoms = len(atoms)
            volume = atoms.get_volume()
            
            if atoms.cell.any():
                cell_params = atoms.cell.cellpar()
                a, b, c = cell_params[0], cell_params[1], cell_params[2]
                alpha, beta, gamma = cell_params[3], cell_params[4], cell_params[5]
            else:
                a = b = c = alpha = beta = gamma = np.nan
            
            data.append({
                'structure_id': i,
                'formula': formula,
                'n_atoms': n_atoms,
                'volume': volume,
                'a': a, 'b': b, 'c': c,
                'alpha': alpha, 'beta': beta, 'gamma': gamma,
                'energy': energy
            })
        
        reference_df = pd.DataFrame(data)
        logger.info(f"Created reference DataFrame with {len(reference_df)} structures")
        return reference_df, rmse 
