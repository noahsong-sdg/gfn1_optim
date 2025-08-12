"""
Bulk materials calculator for processing multiple structures from XYZ files.
Handles energy calculations for optimization against reference data.
"""

import numpy as np
import pandas as pd
import ase.io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ase import Atoms
import logging

from calculators.tblite_ase_calculator import TBLiteASECalculator
from config import SystemConfig

logger = logging.getLogger(__name__)

class BulkCalculator:
    """Calculator for bulk materials optimization"""
    
    def __init__(self, param_file: str, method: str = "gfn1", 
                 electronic_temperature: float = 400.0, charge: float = 0.0, spin: int = 0):
        """
        Initialize bulk calculator
        
        Args:
            param_file: Path to TOML parameter file
            method: TBLite method
            electronic_temperature: Electronic temperature in K
            charge: Total charge
            spin: Number of unpaired electrons
        """
        self.param_file = param_file
        self.method = method
        self.electronic_temperature = electronic_temperature
        self.charge = charge
        self.spin = spin
        
    def load_structures_from_xyz(self, xyz_file: str, max_structures: Optional[int] = None) -> List[Atoms]:
        """Load structures from XYZ file"""
        logger.info(f"Loading structures from {xyz_file}")
        structures = list(ase.io.read(xyz_file, index=':'))
        
        if max_structures:
            structures = structures[:max_structures]
            
        logger.info(f"Loaded {len(structures)} structures")
        return structures
    
    def extract_reference_energies(self, structures: List[Atoms]) -> List[float]:
        """Extract reference energies from structure properties"""
        energies = []
        for i, atoms in enumerate(structures):
            # Try to get energy from atoms.info or properties
            if hasattr(atoms, 'info') and 'energy' in atoms.info:
                energy = atoms.info['energy']
            elif hasattr(atoms, 'get_potential_energy'):
                # If already calculated
                energy = atoms.get_potential_energy()
            else:
                logger.warning(f"No energy found for structure {i}, skipping")
                continue
            energies.append(energy)
        
        logger.info(f"Extracted {len(energies)} reference energies")
        return energies
    
    def calculate_energies(self, structures: List[Atoms]) -> List[float]:
        """Calculate energies for all structures using TBLite"""
        energies = []
        failed = 0
        
        for i, atoms in enumerate(structures):
            try:
                # Create calculator for this structure
                calc = TBLiteASECalculator(
                    param_file=self.param_file,
                    method=self.method,
                    electronic_temperature=self.electronic_temperature,
                    charge=self.charge,
                    spin=self.spin
                )
                
                # Calculate energy
                energy = calc.get_potential_energy(atoms)
                energies.append(energy)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(structures)} structures")
                    
            except Exception as e:
                logger.warning(f"Failed to calculate energy for structure {i}: {e}")
                failed += 1
                energies.append(np.nan)  # Use NaN for failed calculations
        
        logger.info(f"Calculated energies for {len(energies) - failed}/{len(structures)} structures")
        return energies
    
    def create_reference_dataframe(self, structures: List[Atoms], 
                                 reference_energies: List[float]) -> pd.DataFrame:
        """Create reference DataFrame with structure info and energies"""
        data = []
        
        for i, (atoms, energy) in enumerate(zip(structures, reference_energies)):
            if np.isnan(energy):
                continue
                
            # Extract structure information
            formula = atoms.get_chemical_formula()
            n_atoms = len(atoms)
            volume = atoms.get_volume()
            
            # Get cell parameters if available
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
        
        df = pd.DataFrame(data)
        logger.info(f"Created reference DataFrame with {len(df)} structures")
        return df
    
    def compute_energy_loss(self, reference_energies: List[float], 
                          calculated_energies: List[float]) -> float:
        """Compute RMSE loss between reference and calculated energies"""
        # Filter out NaN values
        valid_pairs = [(ref, calc) for ref, calc in zip(reference_energies, calculated_energies) 
                      if not (np.isnan(ref) or np.isnan(calc))]
        
        if not valid_pairs:
            raise ValueError("No valid energy pairs found")
        
        ref_vals, calc_vals = zip(*valid_pairs)
        
        # Convert to numpy arrays
        ref_vals = np.array(ref_vals)
        calc_vals = np.array(calc_vals)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((ref_vals - calc_vals) ** 2))
        
        logger.info(f"Energy RMSE: {rmse:.6f} eV ({len(valid_pairs)} structures)")
        return rmse
    
    def process_bulk_system(self, xyz_file: str, max_structures: Optional[int] = None, 
                           return_structures: bool = False) -> Tuple[pd.DataFrame, float, Optional[List[Atoms]]]:
        """
        Process bulk system: load structures, calculate energies, compute loss
        
        Args:
            xyz_file: Path to XYZ file
            max_structures: Maximum number of structures to process
            return_structures: Whether to return loaded structures
            
        Returns:
            reference_df: DataFrame with reference data
            rmse: Root mean square error
            structures: List of ASE Atoms objects (if return_structures=True)
        """
        # Load structures
        structures = self.load_structures_from_xyz(xyz_file, max_structures)
        
        # Extract reference energies
        reference_energies = self.extract_reference_energies(structures)
        
        # Calculate energies with current parameters
        calculated_energies = self.calculate_energies(structures)
        
        # Compute loss
        rmse = self.compute_energy_loss(reference_energies, calculated_energies)
        
        # Create reference DataFrame
        reference_df = self.create_reference_dataframe(structures, reference_energies)
        
        if return_structures:
            return reference_df, rmse, structures
        else:
            return reference_df, rmse
    
    def test_parameters_on_dataset(self, param_file: str, xyz_file: str, 
                                 max_structures: Optional[int] = None) -> Dict[str, float]:
        """
        Test optimized parameters on a different dataset
        
        Args:
            param_file: Path to optimized parameter file
            xyz_file: Path to test XYZ file
            max_structures: Maximum number of structures to test
            
        Returns:
            Dictionary with test metrics
        """
        # Create temporary calculator with test parameters
        test_calc = BulkCalculator(
            param_file=param_file,
            method=self.method,
            electronic_temperature=self.electronic_temperature,
            charge=self.charge,
            spin=self.spin
        )
        
        # Load and process test structures
        structures = test_calc.load_structures_from_xyz(xyz_file, max_structures)
        reference_energies = test_calc.extract_reference_energies(structures)
        calculated_energies = test_calc.calculate_energies(structures)
        
        # Compute metrics
        rmse = test_calc.compute_energy_loss(reference_energies, calculated_energies)
        
        # Additional metrics
        valid_pairs = [(ref, calc) for ref, calc in zip(reference_energies, calculated_energies) 
                      if not (np.isnan(ref) or np.isnan(calc))]
        
        if valid_pairs:
            ref_vals, calc_vals = zip(*valid_pairs)
            ref_vals = np.array(ref_vals)
            calc_vals = np.array(calc_vals)
            
            mae = np.mean(np.abs(ref_vals - calc_vals))
            max_error = np.max(np.abs(ref_vals - calc_vals))
            mean_error = np.mean(ref_vals - calc_vals)  # Bias
        else:
            mae = max_error = mean_error = np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'bias': mean_error,
            'n_structures': len(valid_pairs)
        }

def create_bulk_calculator(param_file: str, system_config: SystemConfig) -> BulkCalculator:
    """Create bulk calculator from system configuration"""
    return BulkCalculator(
        param_file=param_file,
        method="gfn1",
        electronic_temperature=400.0,
        charge=0.0,
        spin=system_config.spin_multiplicity
    )
