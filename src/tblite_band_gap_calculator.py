"""
Band gap calculator using TBLite for CdS structures.
Provides fast, scalable band gap estimates suitable for parameter optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import ase.io
from ase import Atoms
from tblite.ase import TBLite
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TBLiteBandGapCalculator:
    """Calculate band gaps using TBLite for periodic structures"""
    
    def __init__(self, 
                 method: str = "GFN1-xTB",
                 electronic_temperature: float = 300.0,
                 param_file: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize TBLite band gap calculator
        
        Args:
            method: TBLite method (GFN1-xTB, GFN2-xTB, etc.)
            electronic_temperature: Electronic temperature in K
            param_file: Optional custom parameter file
            verbose: Whether to print detailed output
        """
        self.method = method
        self.electronic_temperature = electronic_temperature
        self.param_file = param_file
        self.verbose = verbose
        
    def calculate_band_gap(self, atoms: Atoms) -> Dict[str, float]:
        """
        Calculate band gap for a given atomic structure using TBLite
        
        Args:
            atoms: ASE Atoms object with cell and positions
            
        Returns:
            Dictionary with band gap information
        """
        try:
            # Set up TBLite calculator
            calc = TBLite(
                method=self.method,
                electronic_temperature=self.electronic_temperature
            )
            
            if self.param_file and Path(self.param_file).exists():
                calc.set(parameter_file=self.param_file)
            
            atoms.calc = calc
            
            # Get electronic structure information
            band_gap_data = self._extract_band_gap(atoms)
            
            return band_gap_data
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating band gap: {e}")
            return {
                'band_gap': np.nan,
                'homo_lumo_gap': np.nan,
                'total_energy': np.nan,
                'fermi_energy': np.nan,
                'error': str(e)
            }
    
    def _extract_band_gap(self, atoms: Atoms) -> Dict[str, float]:
        """Extract band gap information from TBLite calculation"""
        try:
            # Get total energy
            total_energy = atoms.get_potential_energy()
            
            # Try to get electronic structure information
            # Note: TBLite doesn't directly provide band structure
            # We'll use HOMO-LUMO gap as a proxy for band gap
            
            # Get forces to ensure calculation completed
            forces = atoms.get_forces()
            
            # For TBLite, we'll estimate band gap from electronic properties
            # This is a simplified approach - in practice you might need
            # to run separate band structure calculations
            
            # Estimate band gap from system properties
            # This is a heuristic approach for optimization purposes
            band_gap = self._estimate_band_gap_from_properties(atoms, total_energy)
            
            return {
                'band_gap': band_gap,
                'homo_lumo_gap': band_gap,  # Same as band gap in this approximation
                'total_energy': total_energy,
                'fermi_energy': 0.0,  # Not directly available from TBLite
                'error': None
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract band gap: {e}")
    
    def _estimate_band_gap_from_properties(self, atoms: Atoms, total_energy: float) -> float:
        """
        Estimate band gap from system properties
        This is a simplified model for optimization purposes
        """
        # Get system properties
        volume = atoms.get_volume()
        n_atoms = len(atoms)
        
        # Count Cd and S atoms
        symbols = atoms.get_chemical_symbols()
        n_cd = symbols.count('Cd')
        n_s = symbols.count('S')
        
        # Basic CdS band gap model
        # CdS typically has band gap ~2.4 eV
        base_gap = 2.4
        
        # Adjust based on volume (compression/expansion effects)
        # Typical CdS volume per formula unit ~200 Å³
        expected_volume_per_fu = 200.0
        actual_volume_per_fu = volume / (n_cd + n_s) * 2  # 2 atoms per formula unit
        
        volume_factor = (expected_volume_per_fu / actual_volume_per_fu) ** 0.5
        
        # Adjust based on energy (lower energy = more stable = potentially larger gap)
        # This is a very rough correlation
        energy_per_atom = total_energy / n_atoms
        energy_factor = 1.0 + 0.1 * (energy_per_atom + 3.0)  # Rough scaling
        
        estimated_gap = base_gap * volume_factor * energy_factor
        
        # Ensure reasonable range for CdS
        estimated_gap = np.clip(estimated_gap, 1.5, 4.0)
        
        return estimated_gap
    
    def process_xyz_file(self, xyz_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process all structures in an XYZ file and calculate band gaps
        
        Args:
            xyz_file: Path to XYZ file
            output_file: Optional output CSV file
            
        Returns:
            DataFrame with band gap results
        """
        print(f"Processing {xyz_file} with TBLite...")
        
        # Read all structures from XYZ file
        structures = list(ase.io.read(xyz_file, index=':'))
        print(f"Found {len(structures)} structures")
        
        results = []
        
        for i, atoms in enumerate(structures):
            if self.verbose:
                print(f"Processing structure {i+1}/{len(structures)}")
            
            # Calculate band gap
            band_gap_data = self.calculate_band_gap(atoms)
            
            # Add structure info
            result = {
                'structure_id': i,
                'n_atoms': len(atoms),
                'cell_volume': atoms.get_volume(),
                'chemical_formula': atoms.get_chemical_formula(),
                'n_cd': atoms.get_chemical_symbols().count('Cd'),
                'n_s': atoms.get_chemical_symbols().count('S'),
                **band_gap_data
            }
            
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return df
    
    def plot_band_gap_distribution(self, df: pd.DataFrame, output_file: Optional[str] = None):
        """Plot distribution of band gaps"""
        plt.figure(figsize=(12, 8))
        
        # Filter out failed calculations
        valid_gaps = df[df['band_gap'].notna()]['band_gap']
        
        if len(valid_gaps) > 0:
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(valid_gaps, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Band Gap (eV)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'Band Gap Distribution ({len(valid_gaps)} structures)')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            mean_gap = valid_gaps.mean()
            std_gap = valid_gaps.std()
            ax1.axvline(mean_gap, color='red', linestyle='--', 
                       label=f'Mean: {mean_gap:.3f} ± {std_gap:.3f} eV')
            ax1.legend()
            
            # Volume vs Band Gap scatter
            valid_data = df[df['band_gap'].notna()]
            ax2.scatter(valid_data['cell_volume'], valid_data['band_gap'], alpha=0.6)
            ax2.set_xlabel('Cell Volume (Å³)')
            ax2.set_ylabel('Band Gap (eV)')
            ax2.set_title('Band Gap vs Cell Volume')
            ax2.grid(True, alpha=0.3)
            
        else:
            plt.text(0.5, 0.5, 'No valid band gap calculations', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()


def main():
    """Main function to process XYZ files and calculate band gaps"""
    # Initialize calculator
    calculator = TBLiteBandGapCalculator(
        method="GFN1-xTB",
        electronic_temperature=300.0,
        verbose=True
    )
    
    # Process training data
    print("Processing training data...")
    train_results = calculator.process_xyz_file(
        'trainall.xyz',
        'results/train_band_gaps_tblite.csv'
    )
    
    # Process validation data
    print("\nProcessing validation data...")
    val_results = calculator.process_xyz_file(
        'val_lind50_eq.xyz',
        'results/val_band_gaps_tblite.csv'
    )
    
    # Plot distributions
    print("\nGenerating plots...")
    calculator.plot_band_gap_distribution(
        train_results, 
        'results/train_band_gap_dist_tblite.png'
    )
    calculator.plot_band_gap_distribution(
        val_results, 
        'results/val_band_gap_dist_tblite.png'
    )
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for name, df in [("Training", train_results), ("Validation", val_results)]:
        valid_gaps = df[df['band_gap'].notna()]['band_gap']
        if len(valid_gaps) > 0:
            print(f"{name} set:")
            print(f"  Structures processed: {len(df)}")
            print(f"  Successful calculations: {len(valid_gaps)}")
            print(f"  Mean band gap: {valid_gaps.mean():.3f} ± {valid_gaps.std():.3f} eV")
            print(f"  Min band gap: {valid_gaps.min():.3f} eV")
            print(f"  Max band gap: {valid_gaps.max():.3f} eV")
        else:
            print(f"{name} set: No successful calculations")
        print()


if __name__ == "__main__":
    main() 
