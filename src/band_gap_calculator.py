"""
Band gap calculator for CdS structures using low-level DFT theory.
Computes band gaps at the gamma point for each structure in XYZ files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import ase.io
from ase import Atoms
from pyscf import gto, dft, scf
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import scf as pbc_scf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class BandGapCalculator:
    """Calculate band gaps at gamma point for periodic structures"""
    
    def __init__(self, 
                 functional: str = 'PBE',
                 basis: str = 'gth-dzvp',
                 kpts: Optional[np.ndarray] = None,
                 verbose: int = 0):
        """
        Initialize band gap calculator
        
        Args:
            functional: DFT functional (PBE, PBE0, etc.)
            basis: Basis set for periodic calculations
            kpts: k-points grid, defaults to gamma point only
            verbose: PySCF verbosity level
        """
        self.functional = functional
        self.basis = basis
        self.kpts = kpts if kpts is not None else np.array([[0, 0, 0]])
        self.verbose = verbose
        
    def calculate_band_gap(self, atoms: Atoms) -> Dict[str, float]:
        """
        Calculate band gap for a given atomic structure
        
        Args:
            atoms: ASE Atoms object with cell and positions
            
        Returns:
            Dictionary with band gap information
        """
        try:
            # Convert ASE atoms to PySCF cell
            cell = self._ase_to_pyscf_cell(atoms)
            
            # Perform SCF calculation
            mf = self._run_scf(cell)
            
            # Calculate band structure at gamma point
            band_gap = self._compute_band_gap(mf)
            
            return band_gap
            
        except Exception as e:
            print(f"Error calculating band gap: {e}")
            return {
                'band_gap': np.nan,
                'vbm': np.nan,
                'cbm': np.nan,
                'is_direct': False,
                'error': str(e)
            }
    
    def _ase_to_pyscf_cell(self, atoms: Atoms) -> pbc_gto.Cell:
        """Convert ASE Atoms to PySCF Cell object"""
        # Get cell parameters
        cell = pbc_gto.Cell()
        cell.a = atoms.cell
        cell.atom = []
        
        # Add atoms with positions
        for i, symbol in enumerate(atoms.get_chemical_symbols()):
            pos = atoms.get_positions()[i]
            cell.atom.append([symbol, pos])
        
        # Set basis and pseudo
        cell.basis = self.basis
        cell.pseudo = 'gth-pbe'
        cell.verbose = self.verbose
        cell.build()
        
        return cell
    
    def _run_scf(self, cell: pbc_gto.Cell) -> pbc_scf.KRKS:
        """Run SCF calculation"""
        # Use K-point SCF for periodic systems
        mf = pbc_dft.KRKS(cell, kpts=self.kpts)
        mf.xc = self.functional
        mf.verbose = self.verbose
        
        # Run SCF
        mf.kernel()
        
        return mf
    
    def _compute_band_gap(self, mf: pbc_scf.KRKS) -> Dict[str, float]:
        """Compute band gap from converged SCF"""
        # Get eigenvalues at gamma point (k=0)
        mo_energy = mf.mo_energy[0]  # First k-point is gamma
        
        # Find HOMO and LUMO
        n_occ = mf.cell.nelectron // 2  # Assuming closed shell
        
        # Get occupied and unoccupied levels
        occupied = mo_energy[:n_occ]
        unoccupied = mo_energy[n_occ:]
        
        # VBM and CBM
        vbm = np.max(occupied)
        cbm = np.min(unoccupied)
        
        # Band gap
        band_gap = cbm - vbm
        
        # Check if direct (both VBM and CBM at gamma)
        is_direct = True  # Since we're only at gamma point
        
        return {
            'band_gap': band_gap * 27.2114,  # Convert to eV
            'vbm': vbm * 27.2114,
            'cbm': cbm * 27.2114,
            'is_direct': is_direct,
            'error': None
        }
    
    def process_xyz_file(self, xyz_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process all structures in an XYZ file and calculate band gaps
        
        Args:
            xyz_file: Path to XYZ file
            output_file: Optional output CSV file
            
        Returns:
            DataFrame with band gap results
        """
        print(f"Processing {xyz_file}...")
        
        # Read all structures from XYZ file
        structures = list(ase.io.read(xyz_file, index=':'))
        print(f"Found {len(structures)} structures")
        
        results = []
        
        for i, atoms in enumerate(structures):
            print(f"Processing structure {i+1}/{len(structures)}")
            
            # Calculate band gap
            band_gap_data = self.calculate_band_gap(atoms)
            
            # Add structure info
            result = {
                'structure_id': i,
                'n_atoms': len(atoms),
                'cell_volume': atoms.get_volume(),
                'chemical_formula': atoms.get_chemical_formula(),
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
        plt.figure(figsize=(10, 6))
        
        # Filter out failed calculations
        valid_gaps = df[df['band_gap'].notna()]['band_gap']
        
        if len(valid_gaps) > 0:
            plt.hist(valid_gaps, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Band Gap (eV)')
            plt.ylabel('Frequency')
            plt.title(f'Band Gap Distribution ({len(valid_gaps)} structures)')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_gap = valid_gaps.mean()
            std_gap = valid_gaps.std()
            plt.axvline(mean_gap, color='red', linestyle='--', 
                       label=f'Mean: {mean_gap:.3f} ± {std_gap:.3f} eV')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No valid band gap calculations', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()


def main():
    """Main function to process XYZ files and calculate band gaps"""
    # Initialize calculator with low-level settings
    calculator = BandGapCalculator(
        functional='PBE',
        basis='gth-dzvp',
        verbose=0
    )
    
    # Process training data
    print("Processing training data...")
    train_results = calculator.process_xyz_file(
        'trainall.xyz',
        'results/train_band_gaps.csv'
    )
    
    # Process validation data
    print("\nProcessing validation data...")
    val_results = calculator.process_xyz_file(
        'val_lind50_eq.xyz',
        'results/val_band_gaps.csv'
    )
    
    # Plot distributions
    print("\nGenerating plots...")
    calculator.plot_band_gap_distribution(
        train_results, 
        'results/train_band_gap_dist.png'
    )
    calculator.plot_band_gap_distribution(
        val_results, 
        'results/val_band_gap_dist.png'
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
