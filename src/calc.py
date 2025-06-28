"""
General calculator for different molecular/material systems.
Uses system configuration to handle H2, Si2, CdS, etc. in a unified way.
"""

from pyscf import gto, scf, cc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from ase import Atoms
from tblite.ase import TBLite
import tempfile
import subprocess
import os
from typing import Optional, Union, List, Tuple
from enum import Enum
from dataclasses import dataclass

from config import SystemConfig, SystemType, CalculationType, get_system_config
from config import get_calculation_distances, create_molecule_geometry, get_isolated_atom_symbol

# Portable paths
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

class CalcMethod(Enum):
    CCSD = "ccsd"
    GFN1_XTB = "gfn1-xtb"
    XTB_CUSTOM = "custom_tblite"

@dataclass 
class CalcConfig:
    method: CalcMethod 
    basis: str = "cc-pVTZ"
    elec_temp: float = 300.0
    param_file: Optional[str] = None 
    spin: int = 0

class GeneralCalculator:
    """General calculator that can handle different molecular systems"""
    
    def __init__(self, calc_config: CalcConfig, system_config: SystemConfig):
        self.calc_config = calc_config
        self.system_config = system_config
        
        # Override spin multiplicity from system config if not explicitly set
        if calc_config.spin == 0 and hasattr(system_config, 'spin_multiplicity'):
            self.calc_config.spin = system_config.spin_multiplicity

    def calculate_energy(self, atoms) -> float:
        """Calculate energy for given atomic configuration"""
        print(f"Calculating energy with {self.calc_config.method.value}...")
        
        if self.calc_config.method == CalcMethod.CCSD:
            return self._calc_ccsd(atoms)
        elif self.calc_config.method == CalcMethod.GFN1_XTB:
            return self._calc_xtb(atoms)
        elif self.calc_config.method == CalcMethod.XTB_CUSTOM:
            return self._calc_custom_tblite(atoms)
        else:
            raise ValueError(f"Unsupported method: {self.calc_config.method}")

    def _calc_ccsd(self, atoms) -> float:
        """CCSD calculation using PySCF"""
        mol = gto.Mole()

        if isinstance(atoms, str):
            # Single atom
            mol.atom = f'{atoms} 0 0 0'
            mol.spin = self._get_atom_spin_multiplicity(atoms)
        else:
            # Molecule
            atom_str = '\n'.join([f"{sym} {pos[0]} {pos[1]} {pos[2]}" for sym, pos in atoms])
            mol.atom = atom_str
            mol.spin = self.calc_config.spin

        mol.basis = self.calc_config.basis
        mol.symmetry = False
        mol.build()

        mf = scf.UHF(mol)
        mf.kernel()

        ccsd = cc.CCSD(mf)
        e_corr, t1, t2 = ccsd.kernel()
        return mf.e_tot + e_corr
    
    def _calc_xtb(self, atoms) -> float:
        """GFN1-xTB calculation using TBLite/ASE"""
        if isinstance(atoms, str):
            # Single atom
            atoms_obj = Atoms(atoms, positions=[(0, 0, 0)])
            spin = self._get_atom_spin_multiplicity(atoms)
        else:
            # Molecule
            symbols = [atom[0] for atom in atoms]
            positions = [atom[1] for atom in atoms]
            atoms_obj = Atoms(symbols, positions=positions)
            spin = self.calc_config.spin

        atoms_obj.calc = TBLite(method="GFN1-xTB", electronic_temperature=self.calc_config.elec_temp)
        if spin > 0: 
            atoms_obj.calc.set(spin=spin)

        return atoms_obj.get_potential_energy() * 0.0367493  # eV to Hartree conversion
    
    def _calc_custom_tblite(self, atoms) -> float:
        """Custom TBLite calculation with parameter file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_file = self._create_coord_file(tmpdir, atoms)
            energy = self._run_tblite_subprocess(tmpdir, coord_file)
            if energy is None:
                raise RuntimeError("Failed to calculate energy with TBLite")
            return energy
        
    def _create_coord_file(self, tmpdir: str, atoms) -> str:
        """Create coordinate file for TBLite"""
        coord_file = os.path.join(tmpdir, "coord")

        with open(coord_file, 'w') as f:
            f.write("$coord\n")
            
            if isinstance(atoms, str):
                # Single atom
                f.write(f"0.0 0.0 0.0 {atoms.lower()}\n")
            else:
                # Molecule
                for symbol, (x, y, z) in atoms:
                    # Convert Angstrom to Bohr for TBLite
                    x_bohr = x * 1.88973
                    y_bohr = y * 1.88973  
                    z_bohr = z * 1.88973
                    f.write(f"{x_bohr:.10f} {y_bohr:.10f} {z_bohr:.10f} {symbol.lower()}\n")
                    
            f.write("$end\n")
            
            # Add spin multiplicity for TBLite
            if isinstance(atoms, str):
                spin = self._get_atom_spin_multiplicity(atoms)
                multiplicity = spin + 1
            else:
                multiplicity = self.calc_config.spin + 1
            
            if multiplicity > 1:
                f.write(f"$spin\n{multiplicity}\n$end\n")
                
        return coord_file
    
    def _run_tblite_subprocess(self, tmpdir: str, coord_file: str) -> float:
        """Run TBLite subprocess with custom parameters"""
        if not self.calc_config.param_file or not Path(self.calc_config.param_file).exists():
            raise FileNotFoundError(f"Parameter file {self.calc_config.param_file} not found")
        
        param_file_abs = Path(self.calc_config.param_file).resolve()
        cmd = f"tblite run --method gfn1 --param {param_file_abs} {coord_file}"
        result = subprocess.run(
            cmd.split(),
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=False)
        
        if result.returncode != 0:
            error_msg = f"TBLite command failed with exit code {result.returncode}\n"
            error_msg += f"Command: {cmd}\n"
            error_msg += f"Working directory: {tmpdir}\n"
            error_msg += f"STDOUT:\n{result.stdout}\n"
            error_msg += f"STDERR:\n{result.stderr}"
            raise RuntimeError(error_msg)
        
        return self._parse_tblite_e(result.stdout)

    def _parse_tblite_e(self, stdout: str) -> float:
        """Parse energy from TBLite output"""
        # Find all lines containing "total energy" and take the LAST one
        energy_lines = []
        for line in stdout.split('\n'):
            if "total energy" in line.lower():
                energy_lines.append(line)
        
        if energy_lines:
            final_line = energy_lines[-1]
            parts = final_line.split()
            for i in [-1, -2, -3]:
                try:
                    energy = float(parts[i])
                    return energy
                except (ValueError, IndexError):
                    continue
        
        raise ValueError(f"No energy found in TBLite output. Output was:\n{stdout}")
    
    def _get_atom_spin_multiplicity(self, element: str) -> int:
        """Get spin multiplicity for isolated atoms"""
        # Ground state spin multiplicities for common elements
        spin_multiplicities = {
            'H': 1,   # 2S+1 = 2, so S = 1/2, spin = 1
            'C': 2,   # 2S+1 = 3, so S = 1, spin = 2  
            'N': 3,   # 2S+1 = 4, so S = 3/2, spin = 3
            'O': 2,   # 2S+1 = 3, so S = 1, spin = 2
            'Si': 2,  # 2S+1 = 3, so S = 1, spin = 2
            'Cd': 0,  # Singlet
            'S': 2,   # Triplet
            'Zn': 0,  # Singlet 
            'Ga': 1,  # Doublet
        }
        return spin_multiplicities.get(element, 0)

class DissociationCurveGenerator:
    """Generate dissociation curves for diatomic molecules"""
    
    def __init__(self, calculator: GeneralCalculator):
        self.calculator = calculator
        
    def generate_curve(self, distances: Optional[np.ndarray] = None, save: bool = True, 
                      filename: Optional[str] = None) -> pd.DataFrame:
        """Generate dissociation curve for the configured system"""
        
        system_config = self.calculator.system_config
        
        if system_config.system_type != SystemType.DIATOMIC_MOLECULE:
            raise ValueError(f"Dissociation curves only for diatomic molecules, not {system_config.system_type}")
        
        if distances is None:
            distances = get_calculation_distances(system_config)
        
        print(f"Calculating {system_config.name} curve with {self.calculator.calc_config.method.value}...")
        
        # Calculate reference: 2 isolated atoms
        atom_symbol = get_isolated_atom_symbol(system_config)
        atom_energy = self.calculator.calculate_energy(atom_symbol)
        print(f"{atom_symbol} atom energy: {atom_energy:.8f} Hartree")
        
        energies = []
        
        for distance in distances:
            # Molecule at given distance
            molecule_atoms = create_molecule_geometry(system_config, distance)
            
            molecule_energy = self.calculator.calculate_energy(molecule_atoms)
            dissociation_energy = molecule_energy - 2.0 * atom_energy
            energies.append(dissociation_energy)
            print(f"  Distance {distance:.2f} Å: {dissociation_energy:.8f} Hartree")
        
        df = pd.DataFrame({
            'Distance': distances,
            'Energy': energies
        })
        
        if save and filename:
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filename, index=False)
            print(f"Saved to {filename}")
            
        return df

class GeneralStudyManager:
    """High-level manager for molecular/material studies"""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.system_config = get_system_config(system_name)
        self.results = {}
        
    def add_method(self, name: str, calc_config: CalcConfig, filename: Optional[str] = None) -> None:
        """Add a calculation method to the study"""
        if filename and Path(filename).exists():
            print(f"Loading existing data for {name} from {filename}")
            self.results[name] = pd.read_csv(filename)
        else:
            calculator = GeneralCalculator(calc_config, self.system_config)
            
            if self.system_config.calculation_type == CalculationType.DISSOCIATION_CURVE:
                generator = DissociationCurveGenerator(calculator)
                self.results[name] = generator.generate_curve(
                    save=bool(filename), 
                    filename=filename
                )
            else:
                raise NotImplementedError(f"Calculation type {self.system_config.calculation_type} not implemented yet")
    
    def calculate_rmse_vs_reference(self, reference_method: str) -> None:
        """Calculate RMSE compared to reference method"""
        if reference_method not in self.results:
            print(f"Warning: Reference method '{reference_method}' not found in results")
            return
        
        ref_data = self.results[reference_method]
        ref_energies = ref_data['Energy'].values
        
        # Find minimum for relative energies
        min_idx = np.argmin(ref_energies)
        min_energy = ref_energies[min_idx]
        
        if 'Distance' in ref_data.columns:
            distances = ref_data['Distance'].values
            min_distance = distances[min_idx]
            print(f"{reference_method} minimum: {min_energy:.8f} Hartree at {min_distance:.2f} Å")
        else:
            print(f"{reference_method} minimum: {min_energy:.8f} Hartree")
        
        ref_relative = ref_energies - min_energy
        
        for name, data in self.results.items():
            if name == reference_method:
                continue
                
            if isinstance(data, pd.DataFrame):
                calc_energies = data['Energy'].values
            else:
                calc_energies = data
            
            calc_min_idx = np.argmin(calc_energies)
            calc_min = calc_energies[calc_min_idx]
            
            if 'Distance' in data.columns:
                calc_distances = data['Distance'].values
                calc_min_distance = calc_distances[calc_min_idx]
                print(f"{name} minimum: {calc_min:.8f} Hartree at {calc_min_distance:.2f} Å")
            else:
                print(f"{name} minimum: {calc_min:.8f} Hartree")

            # Calculate relative energies
            calc_relative = calc_energies - calc_min
                
            rmse_rel = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            mae_rel = np.mean(np.abs(ref_relative - calc_relative))
            max_error_rel = np.max(np.abs(ref_relative - calc_relative))
            
            print(f"{name}:")
            print(f"  RMSE:      {rmse_rel:.6f} Hartree")
            print(f"  MAE:       {mae_rel:.6f} Hartree")
            print(f"  Max Error: {max_error_rel:.6f} Hartree")

    def plot_comparison(self, output_file: Optional[str] = None) -> None:
        """Create comparison plot of all methods"""
        if output_file is None:
            output_file = f'results/plots/{self.system_name}_comparison.png'
            
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        x_label = 'Distance (Å)'
        if self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
            x_label = 'Lattice Parameter (Å)'
        
        for i, (name, data) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            
            if isinstance(data, pd.DataFrame):
                energies = data['Energy'].values
                if 'Distance' in data.columns:
                    x_values = data['Distance'].values
                else:
                    x_values = np.arange(len(energies))
            else:
                energies = data
                x_values = np.arange(len(energies))
            
            # Convert to relative energies (subtract minimum)
            min_energy = np.min(energies)
            relative_energies = energies - min_energy
                
            plt.plot(x_values, relative_energies, color=color, linewidth=2, 
                    label=name, marker='o', markersize=3, alpha=0.8)
            
            # Print minimum info
            min_idx = np.argmin(energies)
            if len(x_values) > min_idx:
                min_x = x_values[min_idx]
                print(f"{name} minimum: {min_energy:.8f} Hartree at {min_x:.3f}")
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Relative Energy (Hartree)', fontsize=12)
        plt.title(f'{self.system_name} Potential Energy Curves Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved as {output_file}")

def main():
    """Example usage with different systems"""
    
    # Test with Si2
    print("Testing Si2 system:")
    print("=" * 30)

    study = GeneralStudyManager("Si2")

    # Add CCSD calculation
    study.add_method("CCSD", CalcConfig(method=CalcMethod.CCSD), "results/curves/si_ccsd_500.csv")

    # Add GFN1-xTB calculation
    xtb_config = CalcConfig(method=CalcMethod.GFN1_XTB)
    study.add_method("GFN1-xTB", xtb_config, "results/curves/si_pure_500.csv")
    
    # Plot results
    study.plot_comparison()

if __name__ == "__main__":
    main()
