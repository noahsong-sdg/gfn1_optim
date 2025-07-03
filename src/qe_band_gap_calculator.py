"""
Band gap calculator using Quantum ESPRESSO for CdS structures.
Provides high-quality reference band gaps for parameter optimization.
Designed for HPC environments with proper scalability.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import ase.io
from ase import Atoms
import subprocess
import tempfile
import os
import re
import warnings
warnings.filterwarnings('ignore')

class QEBandGapCalculator:
    """Calculate band gaps using Quantum ESPRESSO for periodic structures"""
    
    def __init__(self, 
                 functional: str = 'PBE',
                 ecutwfc: float = 40.0,
                 ecutrho: float = 160.0,
                 kpts: Optional[List[int]] = None,
                 nbands: Optional[int] = None,
                 occupations: str = 'smearing',
                 smearing: str = 'gaussian',
                 degauss: float = 0.01,
                 conv_thr: float = 1e-6,
                 mixing_beta: float = 0.7,
                 max_seconds: int = 3600,
                 verbose: bool = False):
        """
        Initialize QE band gap calculator
        
        Args:
            functional: DFT functional (PBE, PBE0, HSE06, etc.)
            ecutwfc: Wavefunction cutoff in Ry
            ecutrho: Charge density cutoff in Ry
            kpts: k-points grid [nx, ny, nz]
            nbands: Number of bands to calculate
            occupations: Occupation scheme
            smearing: Smearing type
            degauss: Smearing width in Ry
            conv_thr: Convergence threshold
            mixing_beta: Mixing parameter
            max_seconds: Maximum calculation time
            verbose: Whether to print detailed output
        """
        self.functional = functional
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.kpts = kpts if kpts is not None else [2, 2, 2]
        self.nbands = nbands
        self.occupations = occupations
        self.smearing = smearing
        self.degauss = degauss
        self.conv_thr = conv_thr
        self.mixing_beta = mixing_beta
        self.max_seconds = max_seconds
        self.verbose = verbose
        
    def calculate_band_gap(self, atoms: Atoms) -> Dict[str, float]:
        """
        Calculate band gap for a given atomic structure using QE
        
        Args:
            atoms: ASE Atoms object with cell and positions
            
        Returns:
            Dictionary with band gap information
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create QE input files
                self._create_qe_input(atoms, tmpdir)
                
                # Run QE calculation
                success = self._run_qe_calculation(tmpdir)
                
                if not success:
                    raise RuntimeError("QE calculation failed")
                
                # Parse results
                band_gap_data = self._parse_qe_output(tmpdir)
                
                return band_gap_data
                
        except Exception as e:
            if self.verbose:
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
    
    def _create_qe_input(self, atoms: Atoms, tmpdir: str):
        """Create Quantum ESPRESSO input files"""
        
        # Determine number of bands if not specified
        if self.nbands is None:
            n_electrons = sum(self._get_atomic_number(symbol) for symbol in atoms.get_chemical_symbols())
            self.nbands = max(n_electrons // 2 + 10, 50)  # At least 50 bands
        
        # Create system name
        system_name = "cds_system"
        
        # Write SCF input
        scf_input = f"""&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = '{system_name}'
    pseudo_dir = './'
    outdir = './'
    max_seconds = {self.max_seconds}
/

&SYSTEM
    ibrav = 0
    nat = {len(atoms)}
    ntyp = {len(set(atoms.get_chemical_symbols()))}
    ecutwfc = {self.ecutwfc}
    ecutrho = {self.ecutrho}
    nbnd = {self.nbands}
    occupations = '{self.occupations}'
    smearing = '{self.smearing}'
    degauss = {self.degauss}
    nspin = 1
/

&ELECTRONS
    conv_thr = {self.conv_thr}
    mixing_beta = {self.mixing_beta}
    diagonalization = 'david'
/

ATOMIC_SPECIES
"""
        
        # Add atomic species
        unique_symbols = list(set(atoms.get_chemical_symbols()))
        for symbol in unique_symbols:
            atomic_number = self._get_atomic_number(symbol)
            mass = self._get_atomic_mass(symbol)
            scf_input += f"{symbol} {mass:.3f} {symbol.lower()}.UPF\n"
        
        # Add cell parameters
        scf_input += "\nCELL_PARAMETERS angstrom\n"
        cell = atoms.cell
        for i in range(3):
            scf_input += f"{cell[i, 0]:.8f} {cell[i, 1]:.8f} {cell[i, 2]:.8f}\n"
        
        # Add k-points
        scf_input += f"\nK_POINTS automatic\n{self.kpts[0]} {self.kpts[1]} {self.kpts[2]} 0 0 0\n"
        
        # Add atomic positions
        scf_input += "\nATOMIC_POSITIONS angstrom\n"
        for i, symbol in enumerate(atoms.get_chemical_symbols()):
            pos = atoms.get_positions()[i]
            scf_input += f"{symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n"
        
        # Write input file
        with open(os.path.join(tmpdir, f"{system_name}.scf.in"), 'w') as f:
            f.write(scf_input)
        
        # Create band structure input for gamma point
        bands_input = f"""&CONTROL
    calculation = 'bands'
    restart_mode = 'restart'
    prefix = '{system_name}'
    pseudo_dir = './'
    outdir = './'
/

&SYSTEM
    ibrav = 0
    nat = {len(atoms)}
    ntyp = {len(set(atoms.get_chemical_symbols()))}
    ecutwfc = {self.ecutwfc}
    ecutrho = {self.ecutrho}
    nbnd = {self.nbands}
    nspin = 1
/

&ELECTRONS
    diagonalization = 'david'
/

ATOMIC_SPECIES
"""
        
        # Add atomic species again
        for symbol in unique_symbols:
            atomic_number = self._get_atomic_number(symbol)
            mass = self._get_atomic_mass(symbol)
            bands_input += f"{symbol} {mass:.3f} {symbol.lower()}.UPF\n"
        
        # Add cell parameters
        bands_input += "\nCELL_PARAMETERS angstrom\n"
        for i in range(3):
            bands_input += f"{cell[i, 0]:.8f} {cell[i, 1]:.8f} {cell[i, 2]:.8f}\n"
        
        # Add k-points for band structure (gamma point only)
        bands_input += "\nK_POINTS crystal\n1\n0.0 0.0 0.0 1.0\n"
        
        # Add atomic positions
        bands_input += "\nATOMIC_POSITIONS angstrom\n"
        for i, symbol in enumerate(atoms.get_chemical_symbols()):
            pos = atoms.get_positions()[i]
            bands_input += f"{symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n"
        
        # Write bands input file
        with open(os.path.join(tmpdir, f"{system_name}.bands.in"), 'w') as f:
            f.write(bands_input)
    
    def _run_qe_calculation(self, tmpdir: str) -> bool:
        """Run Quantum ESPRESSO calculations"""
        try:
            # Run SCF calculation
            scf_cmd = ["pw.x", "-in", "cds_system.scf.in", "-out", "cds_system.scf.out"]
            result = subprocess.run(
                scf_cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=self.max_seconds + 60
            )
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"SCF calculation failed: {result.stderr}")
                return False
            
            # Run bands calculation
            bands_cmd = ["pw.x", "-in", "cds_system.bands.in", "-out", "cds_system.bands.out"]
            result = subprocess.run(
                bands_cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=self.max_seconds + 60
            )
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"Bands calculation failed: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                print("QE calculation timed out")
            return False
        except Exception as e:
            if self.verbose:
                print(f"Error running QE: {e}")
            return False
    
    def _parse_qe_output(self, tmpdir: str) -> Dict[str, float]:
        """Parse QE output to extract band gap information"""
        try:
            # Read SCF output for total energy
            scf_output = os.path.join(tmpdir, "cds_system.scf.out")
            total_energy = self._extract_total_energy(scf_output)
            
            # Read bands output for band structure
            bands_output = os.path.join(tmpdir, "cds_system.bands.out")
            band_gap, vbm, cbm, fermi_energy = self._extract_band_gap(bands_output)
            
            return {
                'band_gap': band_gap,
                'vbm': vbm,
                'cbm': cbm,
                'fermi_energy': fermi_energy,
                'total_energy': total_energy,
                'is_direct': True,  # Since we're only at gamma point
                'error': None
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse QE output: {e}")
    
    def _extract_total_energy(self, output_file: str) -> float:
        """Extract total energy from QE output"""
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Look for total energy
        match = re.search(r'!.*total energy.*=\s*([-\d.]+)\s*Ry', content)
        if match:
            return float(match.group(1)) * 13.6057  # Convert Ry to eV
        
        raise ValueError("Total energy not found in QE output")
    
    def _extract_band_gap(self, output_file: str) -> Tuple[float, float, float, float]:
        """Extract band gap information from QE bands output"""
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Extract eigenvalues at gamma point
        eigenvalues = []
        lines = content.split('\n')
        in_eigenvalues = False
        
        for line in lines:
            if 'k =' in line and '0.0000' in line:  # Gamma point
                in_eigenvalues = True
                continue
            elif in_eigenvalues and 'band energies' in line:
                continue
            elif in_eigenvalues and line.strip() and not line.startswith('k ='):
                # Parse eigenvalue line
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        eigenvalues.append(float(parts[1]))
                    except ValueError:
                        continue
            elif in_eigenvalues and line.startswith('k ='):
                break
        
        if not eigenvalues:
            raise ValueError("No eigenvalues found in QE output")
        
        # Sort eigenvalues
        eigenvalues.sort()
        
        # Find HOMO and LUMO (assuming closed shell)
        # For CdS, we need to count electrons properly
        n_electrons = 48 + 16  # Cd: 48 electrons, S: 16 electrons
        n_occupied = n_electrons // 2
        
        if n_occupied >= len(eigenvalues):
            raise ValueError(f"Not enough bands calculated: {len(eigenvalues)} < {n_occupied}")
        
        vbm = eigenvalues[n_occupied - 1]
        cbm = eigenvalues[n_occupied]
        band_gap = cbm - vbm
        
        # Estimate Fermi energy (mid-gap)
        fermi_energy = (vbm + cbm) / 2
        
        return band_gap, vbm, cbm, fermi_energy
    
    def _get_atomic_number(self, symbol: str) -> int:
        """Get atomic number for element"""
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50
        }
        return atomic_numbers.get(symbol, 0)
    
    def _get_atomic_mass(self, symbol: str) -> float:
        """Get atomic mass for element"""
        atomic_masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
            'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
            'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.380,
            'Ga': 69.723, 'Ge': 72.640, 'As': 74.922, 'Se': 78.960, 'Br': 79.904,
            'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.620, 'Y': 88.906, 'Zr': 91.224,
            'Nb': 92.906, 'Mo': 95.960, 'Tc': 98.000, 'Ru': 101.070, 'Rh': 102.906,
            'Pd': 106.420, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.710
        }
        return atomic_masses.get(symbol, 0.0)
    
    def process_xyz_file(self, xyz_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process all structures in an XYZ file and calculate band gaps
        
        Args:
            xyz_file: Path to XYZ file
            output_file: Optional output CSV file
            
        Returns:
            DataFrame with band gap results
        """
        print(f"Processing {xyz_file} with Quantum ESPRESSO...")
        
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


def main():
    """Main function to process XYZ files and calculate band gaps"""
    # Initialize calculator with HPC-optimized settings
    calculator = QEBandGapCalculator(
        functional='PBE',
        ecutwfc=40.0,
        ecutrho=160.0,
        kpts=[2, 2, 2],
        nbands=100,
        occupations='smearing',
        smearing='gaussian',
        degauss=0.01,
        conv_thr=1e-6,
        mixing_beta=0.7,
        max_seconds=3600,
        verbose=True
    )
    
    # Process training data
    print("Processing training data...")
    train_results = calculator.process_xyz_file(
        'trainall.xyz',
        'results/train_band_gaps_qe.csv'
    )
    
    # Process validation data
    print("\nProcessing validation data...")
    val_results = calculator.process_xyz_file(
        'val_lind50_eq.xyz',
        'results/val_band_gaps_qe.csv'
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
