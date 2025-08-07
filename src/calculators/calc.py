"""
"""

from pyscf import gto, scf, cc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from tblite.ase import TBLite
import tempfile
import subprocess
import os
from typing import Optional, Union, List, Tuple
from enum import Enum
from dataclasses import dataclass

from config import SystemConfig, SystemType, CalculationType, get_system_config
from config import get_calculation_distances, create_molecule_geometry, get_isolated_atom_symbol

# Set up logger
logger = logging.getLogger(__name__)

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

    def calculate_energy(self, atoms) -> float:
        """Calculate energy for given atomic configuration"""
        # Suppress verbose output during optimization
        
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
            mol.spin = self.system_config.spin_multiplicity

        mol.basis = self.calc_config.basis
        mol.symmetry = False
        mol.build()

        mf = scf.UHF(mol)
        mf.max_cycle = 350  # Increase max SCF cycles for better convergence
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
            # Molecule or bulk structure
            if isinstance(atoms, Atoms):
                atoms_obj = atoms
                spin = self.calc_config.spin
            else:
                # Molecule from list of tuples
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
            elif isinstance(atoms, Atoms):
                # ASE Atoms object (could be bulk structure)
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                cell = atoms.get_cell()
                
                # Write cell information if periodic
                if any(atoms.get_pbc()):
                    f.write("$cell\n")
                    for i in range(3):
                        f.write(f"{cell[i, 0]:.10f} {cell[i, 1]:.10f} {cell[i, 2]:.10f}\n")
                    f.write("$end\n")
                    f.write("$periodic 3\n")
                    f.write("$end\n")
                
                # Write atomic positions
                for symbol, pos in zip(symbols, positions):
                    # Convert Angstrom to Bohr for TBLite
                    x_bohr = pos[0] * 1.88973
                    y_bohr = pos[1] * 1.88973  
                    z_bohr = pos[2] * 1.88973
                    f.write(f"{x_bohr:.10f} {y_bohr:.10f} {z_bohr:.10f} {symbol.lower()}\n")
            else:
                # Molecule from list of tuples
                for symbol, (x, y, z) in atoms:
                    # Convert Angstrom to Bohr for TBLite
                    x_bohr = x * 1.88973
                    y_bohr = y * 1.88973  
                    z_bohr = z * 1.88973
                    f.write(f"{x_bohr:.10f} {y_bohr:.10f} {z_bohr:.10f} {symbol.lower()}\n")
                    
            f.write("$end\n")

        return coord_file
    
    def _run_tblite_subprocess(self, tmpdir: str, coord_file: str) -> float:
        """Run TBLite subprocess with custom parameters"""
        if not self.calc_config.param_file or not Path(self.calc_config.param_file).exists():
            raise FileNotFoundError(f"Parameter file {self.calc_config.param_file} not found")
        
        param_file_abs = Path(self.calc_config.param_file).resolve()
        
        # Enhanced SCF convergence parameters for optimized parameter files
        cmd = [
            "tblite", "run",
            "--method", "gfn1",
            "--param", str(param_file_abs),
            "--iterations", "350",  # Increased from default 250
            "--etemp", str(self.calc_config.elec_temp),
            # "--spin", str(self.calc_config.spin),
            # "--charge", str(self.calc_config.charge),
            coord_file
        ]
        
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=False)
        
        if result.returncode != 0:
            # Check if it's a convergence issue
            stderr_lower = result.stderr.lower()
            if "convergence" in stderr_lower or "scf" in stderr_lower or "iterations" in stderr_lower:
                # Try with more relaxed convergence
                logger.warning(f"SCF convergence failed for {param_file_abs.name}, trying with relaxed parameters...")
                cmd_relaxed = [
                    "tblite", "run",
                    "--method", "gfn1",
                    "--param", str(param_file_abs),
                    "--iterations", "1000",  # More iterations
                    "--etemp", str(self.calc_config.elec_temp) + 300,
                    # "--spin", str(self.calc_config.spin),
                    # "--charge", str(self.calc_config.charge),
                    coord_file
                ]
                
                result_relaxed = subprocess.run(
                    cmd_relaxed,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=False)
                
                if result_relaxed.returncode == 0:
                    logger.info(f"Calculation succeeded with relaxed convergence for {param_file_abs.name}")
                    return self._parse_tblite_e(result_relaxed.stdout)
                else:
                    error_msg = f"TBLite command failed with exit code {result_relaxed.returncode}\n"
                    error_msg += f"Command: {' '.join(cmd)}\n"
                    error_msg += f"Working directory: {tmpdir}\n"
                    error_msg += f"Parameter file: {param_file_abs}\n"
                    error_msg += f"STDOUT:\n{result_relaxed.stdout}\n"
                    error_msg += f"STDERR:\n{result_relaxed.stderr}"
                    raise RuntimeError(error_msg)

            # If all attempts failed, raise error with detailed information
            error_msg = f"TBLite command failed with exit code {result.returncode}\n"
            error_msg += f"Command: {' '.join(cmd)}\n"
            error_msg += f"Parameter file: {param_file_abs}\n"
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
            'H': 1,   
            'Si': 2,  
            'Cd': 0,  
            'S': 0,   
            'Zn': 0,  
            'Ga': 1, 
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
        
        # Calculate reference: 2 isolated atoms
        atom_symbol = get_isolated_atom_symbol(system_config)
        atom_energy = self.calculator.calculate_energy(atom_symbol)
        
        energies = []
        
        for distance in distances:
            # Molecule at given distance
            molecule_atoms = create_molecule_geometry(system_config, distance)
            
            molecule_energy = self.calculator.calculate_energy(molecule_atoms)
            dissociation_energy = molecule_energy - 2.0 * atom_energy
            energies.append(dissociation_energy)
        
        df = pd.DataFrame({
            'Distance': distances,
            'Energy': energies
        })
        
        if save and filename:
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filename, index=False)
            
        return df

class CrystalGenerator:
    """Generate lattice constants for solid-state systems"""
    
    def __init__(self, calculator: GeneralCalculator):
        self.calculator = calculator
    def get_bandgap(self) -> float:
        # https://wiki.fysik.dtu.dk/ase/ase/dft/bandgap.html
        from ase.dft.bandgap import bandgap
        gap, p1, p2 = bandgap(self.calculator.atoms, direct=True,kpts=(1, 1, 1))
        return gap
    def getElastic(self) -> float:
        # need to implement
        return 0.0
    def compute_stuff(self, a_guess: Optional[float] = None, 
                                  c_guess: Optional[float] = None) -> pd.DataFrame:        
        system_config = self.calculator.system_config
        
        if system_config.system_type != SystemType.SOLID_STATE:
            raise ValueError(f"Lattice optimization only for solid-state systems")
        
        # Use system defaults if not provided
        if a_guess is None:
            a_guess = system_config.lattice_params["a"]
        if c_guess is None:
            c_guess = system_config.lattice_params["c"]
        
        # Create initial structure
        atoms = bulk(system_config.name, crystalstructure='wurtzite', a=a_guess, c=c_guess)
        
        # Set up calculator based on method
        if self.calculator.calc_config.method == CalcMethod.GFN1_XTB:
            atoms.calc = TBLite(method="GFN1-xTB", electronic_temperature=self.calculator.calc_config.elec_temp)
        elif self.calculator.calc_config.method == CalcMethod.XTB_CUSTOM:
            # Use custom TBLite ASE calculator for parameter optimization
            from calculators.tblite_ase_calculator import TBLiteASECalculator
            atoms.calc = TBLiteASECalculator(
                param_file=self.calculator.calc_config.param_file,
                method="gfn1",
                electronic_temperature=self.calculator.calc_config.elec_temp,
                spin=self.calculator.calc_config.spin
            )
        else:
            raise ValueError(f"Lattice optimization not supported for method {self.calculator.calc_config.method}")
        
        # Optimize cell
        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf)
        opt.run(fmax=0.01)  # Convergence criterion

        df = pd.DataFrame({
            'a': [atoms.cell.cellpar()[0]],
            'b': [atoms.cell.cellpar()[1]],
            'c': [atoms.cell.cellpar()[2]],
            'alpha': [atoms.cell.cellpar()[3]],
            'beta': [atoms.cell.cellpar()[4]],
            'gamma': [atoms.cell.cellpar()[5]],
            'energy': [atoms.get_potential_energy() * 0.0367493],  # Convert to Hartree
            #'bandgap': self.get_bandgap(),
            #'elastic': self.getElastic()
        })
        
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
            elif self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
                generator = CrystalGenerator(calculator)
                self.results[name] = generator.compute_stuff()
            else:
                raise NotImplementedError(f"Calculation type {self.system_config.calculation_type} not implemented yet")
    
    def calculate_rmse_vs_reference(self, reference_method: str) -> None:
        """Calculate RMSE compared to reference method or experimental data"""
        if reference_method == "experimental":
            # Use experimental reference data for solid-state systems
            self._calculate_rmse_vs_experimental()
            return
        
        if reference_method not in self.results:
            print(f"Warning: Reference method '{reference_method}' not found in results")
            return
        
        ref_data = self.results[reference_method]
        
        if self.system_config.calculation_type == CalculationType.DISSOCIATION_CURVE:
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
        
        elif self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
            ref_a = ref_data['a'].iloc[ref_data['Energy'].idxmin()]
            ref_c = ref_data['c'].iloc[ref_data['Energy'].idxmin()]
            ref_energy = ref_data['Energy'].min()
            
            print(f"{reference_method} optimized: a={ref_a:.3f} Å, c={ref_c:.3f} Å, E={ref_energy:.8f} Hartree")
            
            for name, data in self.results.items():
                if name == reference_method:
                    continue
                
                calc_a = data['a'].iloc[data['Energy'].idxmin()]
                calc_c = data['c'].iloc[data['Energy'].idxmin()]
                calc_energy = data['Energy'].min()
                
                print(f"{name} optimized: a={calc_a:.3f} Å, c={calc_c:.3f} Å, E={calc_energy:.8f} Hartree")
                
                # Calculate errors in lattice parameters
                a_error = abs(calc_a - ref_a)
                c_error = abs(calc_c - ref_c)
                energy_error = abs(calc_energy - ref_energy)
                
                print(f"{name} errors:")
                print(f"  Δa:        {a_error:.4f} Å")
                print(f"  Δc:        {c_error:.4f} Å")
                print(f"  ΔE:        {energy_error:.6f} Hartree")

    def _calculate_rmse_vs_experimental(self) -> None:
        """Calculate RMSE compared to experimental lattice constants"""
        #reference data for CdS (wurtzite), mat database
        experimental_data = {
            'a': 4.17,      # Å
            'b': 4.17,      # Å  
            'c': 6.78,      # Å
            'alpha': 90.0,  # degrees
            'beta': 90.0,   # degrees
            'gamma': 120.0, # degrees
            'volume': 102.04 # Å³
        }
        
        print(f"Experimental CdS (wurtzite):")
        print(f"  a={experimental_data['a']:.3f} Å, c={experimental_data['c']:.3f} Å")
        print(f"  Volume={experimental_data['volume']:.2f} Å³")
        
        for name, data in self.results.items():
            # Get optimized lattice constants from calculation
            calc_a = data['a'].iloc[data['Energy'].idxmin()]
            calc_c = data['c'].iloc[data['Energy'].idxmin()]
            calc_energy = data['Energy'].min()
            
            # Calculate volume from lattice constants (wurtzite: V = a²c * sin(120°))
            calc_volume = calc_a**2 * calc_c * np.sin(np.radians(120))
            
            print(f"{name} optimized: a={calc_a:.3f} Å, c={calc_c:.3f} Å, E={calc_energy:.8f} Hartree")
            print(f"{name} volume: {calc_volume:.2f} Å³")
            
            # Calculate errors
            a_error = abs(calc_a - experimental_data['a'])
            c_error = abs(calc_c - experimental_data['c'])
            volume_error = abs(calc_volume - experimental_data['volume'])
            
            # Calculate relative errors (%)
            a_rel_error = (a_error / experimental_data['a']) * 100
            c_rel_error = (c_error / experimental_data['c']) * 100
            volume_rel_error = (volume_error / experimental_data['volume']) * 100
            
            print(f"{name} errors vs experimental:")
            print(f"  Δa:        {a_error:.4f} Å ({a_rel_error:.2f}%)")
            print(f"  Δc:        {c_error:.4f} Å ({c_rel_error:.2f}%)")
            print(f"  ΔV:        {volume_error:.2f} Å³ ({volume_rel_error:.2f}%)")
            print()

    def plot_comparison(self, output_file: Optional[str] = None) -> None:
        """Create comparison plot of all methods"""
        if output_file is None:
            output_file = f'results/plots/{self.system_name}_comparison.png'
            
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        if self.system_config.calculation_type == CalculationType.DISSOCIATION_CURVE:
            x_label = 'Distance (Å)'
            
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
            
            plt.ylabel('Relative Energy (Hartree)', fontsize=12)
            
        elif self.system_config.calculation_type == CalculationType.LATTICE_CONSTANTS:
            # Experimental reference point
            exp_a, exp_c = 4.17, 6.78
            plt.plot(exp_a, exp_c, 'k*', markersize=15, label=f'Experimental (a={exp_a:.3f}, c={exp_c:.3f})', zorder=10)
            
            # Create 2D contour plot for lattice constants
            for i, (name, data) in enumerate(self.results.items()):
                color = colors[i % len(colors)]
                
                # Create pivot table for contour plot
                pivot_data = data.pivot(index='c', columns='a', values='Energy')
                
                # Create contour plot
                a_values = pivot_data.columns.values
                c_values = pivot_data.index.values
                energy_values = pivot_data.values
                
                # Find minimum
                min_idx = np.unravel_index(np.nanargmin(energy_values), energy_values.shape)
                min_a = a_values[min_idx[1]]
                min_c = c_values[min_idx[0]]
                min_energy = energy_values[min_idx]
                
                # Create contour plot
                contour = plt.contour(a_values, c_values, energy_values, levels=20, colors=color, alpha=0.7)
                plt.clabel(contour, inline=True, fontsize=8)
                
                # Mark minimum
                plt.plot(min_a, min_c, 'o', color=color, markersize=8, label=f"{name} (a={min_a:.3f}, c={min_c:.3f})")
                
                print(f"{name} minimum: a={min_a:.3f} Å, c={min_c:.3f} Å, E={min_energy:.8f} Hartree")
            
            x_label = 'Lattice Parameter a (Å)'
            plt.ylabel('Lattice Parameter c (Å)', fontsize=12)
        
        plt.xlabel(x_label, fontsize=12)
        plt.title(f'{self.system_name} Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved as {output_file}")

def main():
    """Generate reference data for any system with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reference data for molecular systems")
    parser.add_argument("--system", default="Si2", choices=["H2", "Si2", "CdS"], 
                       help="System to calculate (default: Si2)")
    parser.add_argument("--method", default="CCSD", choices=["CCSD", "GFN1-xTB"], 
                       help="Calculation method (default: CCSD)")
    parser.add_argument("--output", help="Output filename (auto-generated if not specified)")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plot")
    
    args = parser.parse_args()
    
    study = GeneralStudyManager(args.system)
    
    # Auto-generate filename if not specified
    if args.output is None:
        method_name = args.method.lower().replace("-", "_")
        if args.system == "CdS":
            args.output = f"results/lattice/{args.system.lower()}_{method_name}_data.csv"
        else:
            args.output = f"results/curves/{args.system.lower()}_{method_name}_data.csv"
    
    # Add the requested calculation
    if args.method == "CCSD":
        config = CalcConfig(method=CalcMethod.CCSD, basis="cc-pVTZ")
    elif args.method == "GFN1-xTB":
        config = CalcConfig(method=CalcMethod.GFN1_XTB)
    
    study.add_method(args.method, config, args.output)
    
    if args.plot:
        study.plot_comparison()
    
    print(f"\nCompleted! Data saved to: {args.output}")

if __name__ == "__main__":
    main()
