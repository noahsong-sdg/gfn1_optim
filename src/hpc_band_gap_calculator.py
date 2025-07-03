"""
Band gap calculator optimized for HPC environments.
Uses PySCF with efficient settings for 72-atom CdS systems.
Provides high-quality reference band gaps for parameter optimization.
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
from pyscf.pbc import tools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class HPCBandGapCalculator:
    """Calculate band gaps using PySCF optimized for HPC environments"""
    
    def __init__(self, 
                 functional: str = 'PBE',
                 basis: str = 'gth-dzvp',
                 kpts: Optional[np.ndarray] = None,
                 nbands: Optional[int] = None,
                 verbose: int = 0,
                 max_memory: int = 8000,
                 conv_tol: float = 1e-6,
                 max_cycle: int = 100):
        """
        Initialize HPC band gap calculator
        
        Args:
            functional: DFT functional (PBE, PBE0, etc.)
            basis: Basis set for periodic calculations
            kpts: k-points grid, defaults to gamma point only
            nbands: Number of bands to calculate
            verbose: PySCF verbosity level
            max_memory: Maximum memory in MB
            conv_tol: Convergence tolerance
            max_cycle: Maximum SCF cycles
        """
        self.functional = functional
        self.basis = basis
        self.kpts = kpts if kpts is not None else np.array([[0, 0, 0]])
        self.nbands = nbands
        self.verbose = verbose
        self.max_memory = max_memory
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        
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
            
            # Perform SCF calculation with optimized settings
            mf = self._run_optimized_scf(cell)
            
            # Calculate band structure at gamma point
            band_gap = self._compute_band_gap(mf)
            
            return band_gap
            
        except Exception as e:
            print(f"Error calculating band gap: {e}")
            return {
                'band_gap': np.nan,
                'vbm': np.nan,
                'cbm': np.nan,
                'fermi_energy': np.nan,
                'total_energy': np.nan,
                'is_direct': False,
                'error': str(e)
            }
    
    def _ase_to_pyscf_cell(self, atoms: Atoms) -> pbc_gto.Cell:
        """Convert ASE Atoms to PySCF Cell object with optimized settings"""
        # Get cell parameters
        cell = pbc_gto.Cell()
        cell.a = atoms.cell
        cell.atom = []
        
        # Add atoms with positions
        for i, symbol in enumerate(atoms.get_chemical_symbols()):
            pos = atoms.get_positions()[i]
            cell.atom.append([symbol, pos])
        
        # Set basis and pseudo with optimized settings
        cell.basis = self.basis
        cell.pseudo = 'gth-pbe'
        cell.verbose = self.verbose
        cell.max_memory = self.max_memory
        
        # Optimize cell settings for large systems
        cell.precision = 1e-8  # Slightly relaxed precision for speed
        cell.mesh = [25, 25, 25]  # Coarse mesh for speed
        
        cell.build()
        
        return cell
    
    def _run_optimized_scf(self, cell: pbc_gto.Cell) -> pbc_scf.KRKS:
        """Run SCF calculation with HPC-optimized settings"""
        # Use K-point SCF for periodic systems
        mf = pbc_dft.KRKS(cell, kpts=self.kpts)
        mf.xc = self.functional
        mf.verbose = self.verbose
        
        # Optimize SCF settings for large systems
        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.mix = 0.7  # Mixing parameter for better convergence
        
        # Use efficient diagonalization
        mf.diag = 'david'  # Davidson diagonalization
        mf.level_shift = 0.2  # Level shift for better convergence
        
        # Run SCF
        mf.kernel()
        
        return mf
    
    def _compute_band_gap(self, mf: pbc_scf.KRKS) -> Dict[str, float]:
        """Compute band gap from converged SCF with proper electron counting"""
        # Get eigenvalues at gamma point (k=0)
        mo_energy = mf.mo_energy[0]  # First k-point is gamma
        
        # Count electrons properly for CdS
        n_cd = sum(1 for atom in mf.cell.atom if atom[0] == 'Cd')
        n_s = sum(1 for atom in mf.cell.atom if atom[0] == 'S')
        
        # Cd has 48 electrons, S has 16 electrons
        total_electrons = n_cd * 48 + n_s * 16
        n_occupied = total_electrons // 2  # Assuming closed shell
        
        # Get occupied and unoccupied levels
        occupied = mo_energy[:n_occupied]
        unoccupied = mo_energy[n_occupied:]
        
        # VBM and CBM
        vbm = np.max(occupied)
        cbm = np.min(unoccupied)
        
        # Band gap
        band_gap = cbm - vbm
        
        # Fermi energy (mid-gap approximation)
        fermi_energy = (vbm + cbm) / 2
        
        # Total energy
        total_energy = mf.e_tot * 27.2114  # Convert to eV
        
        return {
            'band_gap': band_gap * 27.2114,  # Convert to eV
            'vbm': vbm * 27.2114,
            'cbm': cbm * 27.2114,
            'fermi_energy': fermi_energy * 27.2114,
            'total_energy': total_energy,
            'is_direct': True,  # Since we're only at gamma point
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
        print(f"Processing {xyz_file} with HPC-optimized PySCF...")
        
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
        plt.figure(figsize=(15, 10))
        
        # Filter out failed calculations
        valid_gaps = df[df['band_gap'].notna()]['band_gap']
        
        if len(valid_gaps) > 0:
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
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
            
            # Energy vs Band Gap scatter
            ax3.scatter(valid_data['total_energy'], valid_data['band_gap'], alpha=0.6)
            ax3.set_xlabel('Total Energy (eV)')
            ax3.set_ylabel('Band Gap (eV)')
            ax3.set_title('Band Gap vs Total Energy')
            ax3.grid(True, alpha=0.3)
            
            # VBM vs CBM scatter
            ax4.scatter(valid_data['vbm'], valid_data['cbm'], alpha=0.6)
            ax4.set_xlabel('VBM (eV)')
            ax4.set_ylabel('CBM (eV)')
            ax4.set_title('VBM vs CBM')
            ax4.grid(True, alpha=0.3)
            
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
    # Initialize calculator with HPC-optimized settings
    calculator = HPCBandGapCalculator(
        functional='PBE',
        basis='gth-dzvp',
        kpts=np.array([[0, 0, 0]]),  # Gamma point only
        nbands=100,
        verbose=0,
        max_memory=8000,  # 8 GB memory limit
        conv_tol=1e-6,
        max_cycle=100
    )
    
    # Process training data
    print("Processing training data...")
    train_results = calculator.process_xyz_file(
        'trainall.xyz',
        'results/train_band_gaps_hpc.csv'
    )
    
    # Process validation data
    print("\nProcessing validation data...")
    val_results = calculator.process_xyz_file(
        'val_lind50_eq.xyz',
        'results/val_band_gaps_hpc.csv'
    )
    
    # Plot distributions
    print("\nGenerating plots...")
    calculator.plot_band_gap_distribution(
        train_results, 
        'results/train_band_gap_dist_hpc.png'
    )
    calculator.plot_band_gap_distribution(
        val_results, 
        'results/val_band_gap_dist_hpc.png'
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
