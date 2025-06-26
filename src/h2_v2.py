"""
compares ccsd, gfn1-xtb, and custom tblite for h2 dissociation curve
default params in gfn1-base.toml
fitted params to ccsd using tblite fit found in fitpar.toml
the command is typed out in pixi bc tblite is in my pixi env

bad code things:
the stdout file from tblite has two instances of "total energy". the first is in a row of a column.
the second has the energy following it. so the code extracts the last instance of "total energy:
to get the energy. 

on fitting parameters:
for h2, the fit took > 10 hours of single core cpu time
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

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results" 
DATA_DIR = PROJECT_ROOT / "data"

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
CUSTOM_PARAM_FILE = CONFIG_DIR / "fitpar.toml"

# Reference data files
CCSD_REFERENCE_500 = RESULTS_DIR / "curves" / "h2_ccsd_500.csv"
XTB_DEFAULT_500 = RESULTS_DIR / "curves" / "h2_xtb_500.csv"
CUSTOM_CURVE_500 = RESULTS_DIR / "curves" / "h2_custom_500.csv"

# Optimized parameter files
PSO_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "pso_optimized_params.toml"
GA_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "ga_optimized_params.toml"
BAYESIAN_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "bayesian_optimized_params.toml"

# Output curve files for optimized methods
PSO_CURVE_DATA = RESULTS_DIR / "curves" / "h2_pso_data.csv"
GA_CURVE_DATA = RESULTS_DIR / "curves" / "h2_ga_data.csv"
BAYESIAN_CURVE_DATA = RESULTS_DIR / "curves" / "h2_bayesian_data.csv"

# Plot output
COMPARISON_PLOT = RESULTS_DIR / "plots" / "h2_comparison.png"

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

class MolecularCalculator:
    def __init__(self, config: CalcConfig):
        self.config = config 

    def calculate_energy(self, atoms) -> float:
        print(f"Calculating energy with {self.config.method.value}...")
        if self.config.method == CalcMethod.CCSD:
            return self._calc_ccsd(atoms)
        elif self.config.method == CalcMethod.GFN1_XTB:
            return self._calc_xtb(atoms)
        elif self.config.method == CalcMethod.XTB_CUSTOM:
            return self._calc_custom_tblite(atoms)
        else:
            raise ValueError(f"Unsupported method: {self.config.method}")

    def _calc_ccsd(self, atoms) -> float:
        mol = gto.Mole()

        if atoms == 'H':
            mol.atom = 'H 0 0 0'
            mol.spin = 1
        else:
            atom_str = '\n'.join([f"{sym} {pos[0]} {pos[1]} {pos[2]}" for sym, pos in atoms])
            mol.atom = atom_str
            mol.spin = self.config.spin

        mol.basis = self.config.basis
        mol.symmetry = False
        mol.build()

        mf = scf.UHF(mol)
        mf.kernel()

        ccsd = cc.CCSD(mf)
        e_corr, t1, t2 =ccsd.kernel()
        return mf.e_tot + e_corr
    
    def _calc_xtb(self, atoms) -> float:
        if atoms == 'H':
            atoms = Atoms('H', positions=[(0, 0, 0)])
            spin = 1
        else:
            symbols = [atom[0] for atom in atoms]
            positions = [atom[1] for atom in atoms]
            atoms = Atoms(symbols, positions=positions)
            spin = self.config.spin

        atoms.calc = TBLite(method="GFN1-xTB", electronic_temperature=self.config.elec_temp)
        if spin > 0: atoms.calc.set(spin=spin)

        return atoms.get_potential_energy() * 0.0367 # HARTRE
    
    def _calc_custom_tblite(self, atoms) -> float:
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_file = self._create_coord_file(tmpdir, atoms)
            energy = self._run_tblite_subprocess(tmpdir, coord_file)
            if energy is None:
                raise RuntimeError("Failed to calculate energy with TBLite")
            return energy
        
    def _create_coord_file(self, tmpdir: str, atoms) -> str:
        coord_file = os.path.join(tmpdir, "coord")

        with open(coord_file, 'w') as f:
            f.write("$coord\n")
            if atoms == 'H':
                f.write("0.0 0.0 0.0 h\n")
            else:
                for symbol, (x, y, z) in atoms:
                    # Convert Angstrom to Bohr for TBLite
                    x_bohr = x * 1.88973
                    y_bohr = y * 1.88973  
                    z_bohr = z * 1.88973
                    f.write(f"{x_bohr:.10f} {y_bohr:.10f} {z_bohr:.10f} {symbol.lower()}\n")
                    
            f.write("$end\n")
        return coord_file
    
    def _run_tblite_subprocess(self, tmpdir: str, coord_file: str) -> float:
        if not self.config.param_file or not Path(self.config.param_file).exists():
            raise FileNotFoundError(f"Parameter file {self.config.param_file} not found")
        
        param_file_abs = Path(self.config.param_file).resolve()
        cmd = f"pixi run tblite run --method gfn1 --param {param_file_abs} {coord_file}"
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
        print(f"TBLite output: {result.stdout}")
        return self._parse_tblite_e(result.stdout)

    def _parse_tblite_e(self, stdout: str) -> float:
        print("DEBUG: TBLite output:")
        print("=" * 50)
        print(stdout)
        print("=" * 50)
        # Find all lines containing "total energy" and take the LAST one
        # (to avoid matching column headers in SCF iteration tables)
        energy_lines = []
        for line in stdout.split('\n'):
            if "total energy" in line.lower():
                energy_lines.append(line)
        
        if energy_lines:
            # Take the last occurrence (likely the final converged energy)
            final_line = energy_lines[-1]
            print(f"DEBUG: Using energy line: {final_line}")
            parts = final_line.split()
            # Try different positions for the energy value
            for i in [-1, -2, -3]:
                try:
                    energy = float(parts[i])
                    print(f"DEBUG: Extracted energy: {energy}")
                    return energy
                except (ValueError, IndexError):
                    continue
        
        raise ValueError(f"No energy found in TBLite output. Found {len(energy_lines)} lines with 'total energy'. Output was:\n{stdout}") 
    
class DissociationCurveGenerator:
    """Generate potential energy curves for molecular dissociation"""
    
    def __init__(self, calculator: MolecularCalculator):
        self.calculator = calculator
        
    def generate_h2_curve(self, distances: np.ndarray, save: bool = True, 
                         filename: Optional[str] = None) -> pd.DataFrame:
        """Generate H2 dissociation curve"""
        print(f"Calculating H2 curve with {self.calculator.config.method.value}...")
        
        # Calculate reference: 2 isolated H atoms
        h_atom_energy = self.calculator.calculate_energy('H')
        energies = []
        
        for distance in distances:
            # H2 molecule at given distance
            h2_atoms = [
                ('H', (0.0, 0.0, 0.0)),
                ('H', (0.0, 0.0, distance))
            ]
            
            h2_energy = self.calculator.calculate_energy(h2_atoms)
            dissociation_energy = h2_energy - 2.0 * h_atom_energy
            energies.append(dissociation_energy)
        
        df = pd.DataFrame({
            'Distance': distances,
            'Energy': energies
        })
        
        if save and filename:
            df.to_csv(filename, index=False)
            print(f"Saved to {filename}")
            
        return df

class H2StudyManager:
    """High-level manager for H2 potential energy surface studies"""
    
    def __init__(self, distances: np.ndarray):
        self.distances = distances
        self.results = {}
        
    def add_method(self, name: str, config: CalcConfig, filename: Optional[str] = None) -> None:
        """Add a calculation method to the study"""
        if filename and Path(filename).exists():
            print(f"Loading existing data for {name} from {filename}")
            self.results[name] = pd.read_csv(filename)
        else:
            calculator = MolecularCalculator(config)
            generator = DissociationCurveGenerator(calculator)
            self.results[name] = generator.generate_h2_curve(
                self.distances, 
                save=bool(filename), 
                filename=filename
            )
    
    def calculate_rmse_vs_ccsd(self, reference_method: str = "CCSD/cc-pVTZ") -> None:
        """Calculate RMSE of each method compared to CCSD reference"""
        if reference_method not in self.results:
            print(f"Warning: Reference method '{reference_method}' not found in results")
            return
        
        min_e = self.results[reference_method]['Energy'].values
        min_d = np.argmin(min_e)
        print(f"CCSD minimum: {min_e[min_d]} Hartree at {self.distances[min_d]} Å")

        ref_data = self.results[reference_method]
        ref_energies = ref_data['Energy'].values
        ref_min = ref_energies[-1]
        ref_relative = ref_energies - ref_min
        
        for name, data in self.results.items():
            if name == reference_method:
                continue
                
            if isinstance(data, pd.DataFrame):
                calc_energies = data['Energy'].values
            else:
                calc_energies = data
            
            calc_min = calc_energies[-1]
            calc_mind_d = np.argmin(calc_energies)
            print(f"{name} minimum: {calc_min} Hartree at {self.distances[calc_mind_d]} Å")

            # Calculate relative energies
            calc_relative = calc_energies - calc_min
                
            rmse_rel = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            mae_rel = np.mean(np.abs(ref_relative - calc_relative))
            max_error_rel = np.max(np.abs(ref_relative - calc_relative))
            
            print(f"{name}:")
            print(f"  RMSE:      {rmse_rel} Hartree")
            print(f"  MAE:       {mae_rel} Hartree ")
            print(f"  Max Error: {max_error_rel} Hartree")

    def plot_comparison(self, output_file: str = 'results/plots/h2_comparison.png') -> None:
        """Create comparison plot of all methods"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, data) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            
            if isinstance(data, pd.DataFrame):
                energies = data['Energy'].values
            else:
                energies = data
            
            # Convert to relative energies (subtract minimum to align curves)
            last_energy = energies[-1]
            relative_energies = energies - last_energy
                
            plt.plot(self.distances, relative_energies, color=color, linewidth=2, 
                    label=name, marker='o', markersize=3, alpha=0.8)
            
            # Print minimum info
            min_distance = self.distances[np.argmin(energies)]
            print(f"{name} minimum: {last_energy:.6f} Hartree at {min_distance:.3f} Å")
        
        plt.xlabel('H-H Distance (Å)', fontsize=12)
        plt.ylabel('Relative Energy (Hartree)', fontsize=12)
        plt.title('H₂ Potential Energy Curves Comparison (Relative to Minimum)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved as {output_file}")
        
        # Automatically calculate RMSE after plotting
        self.calculate_rmse_vs_ccsd()

# Main execution
def main():
    distances = np.linspace(0.4, 4.0, 500)
    study = H2StudyManager(distances)
    
    # Add CCSD reference
    ccsd_config = CalcConfig(
        method=CalcMethod.CCSD,
        basis="cc-pVTZ"
    )
    study.add_method("CCSD/cc-pVTZ", ccsd_config, str(CCSD_REFERENCE_500))
    
    # Add standard GFN1-xTB
    xtb_config = CalcConfig(
        method=CalcMethod.GFN1_XTB,
        spin=1
    )
    study.add_method("GFN1-xTB (default)", xtb_config, str(XTB_DEFAULT_500))
    
    # Add custom TBLite (if available)
    if CUSTOM_PARAM_FILE.exists():
        custom_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=str(CUSTOM_PARAM_FILE),
            spin=1
        )
        study.add_method("Custom TBLite", custom_config, str(CUSTOM_CURVE_500))
    else:
        print(f"Custom parameter file {CUSTOM_PARAM_FILE} not found, skipping custom calculation")
    
    # Add PSO optimized parameters (if available)
    if PSO_OPTIMIZED_PARAMS.exists():
        pso_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=str(PSO_OPTIMIZED_PARAMS),
            spin=1
        )
        study.add_method("PSO Optimized", pso_config, str(PSO_CURVE_DATA))
    
    # Add GA optimized parameters (if available)
    if GA_OPTIMIZED_PARAMS.exists():
        ga_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=str(GA_OPTIMIZED_PARAMS),
            spin=1
        )
        study.add_method("GA Optimized", ga_config, str(GA_CURVE_DATA))
    
    # Add Bayesian optimized parameters (if available)
    if BAYESIAN_OPTIMIZED_PARAMS.exists():
        bayesian_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=str(BAYESIAN_OPTIMIZED_PARAMS),
            spin=1
        )
        study.add_method("Bayesian Optimized", bayesian_config, str(BAYESIAN_CURVE_DATA))
    
    # Generate comparison plot
    study.plot_comparison(str(COMPARISON_PLOT))

if __name__ == "__main__":
    main()
