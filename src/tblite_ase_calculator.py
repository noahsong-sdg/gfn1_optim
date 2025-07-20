import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import logging

logger = logging.getLogger(__name__)

class TBLiteASECalculator(Calculator):
    """ASE Calculator interface for TBLite with custom parameters"""
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, 
                 param_file: str,
                 method: str = "gfn1",
                 electronic_temperature: float = 300.0,
                 charge: float = 0.0,
                 spin: int = 0,
                 **kwargs):
        """
        Initialize TBLite ASE Calculator
        
        Args:
            param_file: Path to TOML parameter file with custom parameters
            method: TBLite method (gfn1, gfn2, ipea1)
            electronic_temperature: Electronic temperature in K
            charge: Total charge of the system
            spin: Number of unpaired electrons
        """
        Calculator.__init__(self, **kwargs)
        
        self.param_file = Path(param_file)
        if not self.param_file.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_file}")
        
        self.method = method
        self.electronic_temperature = electronic_temperature
        self.charge = charge
        self.spin = spin
        
    def calculate(self, atoms: Optional[Atoms] = None, 
                 properties: list = ['energy'], 
                 system_changes: list = all_changes):
        """Calculate energy and forces using TBLite"""
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Create temporary directory for calculation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write coordinates to file
            coord_file = Path(tmpdir) / "coord.xyz"
            self._write_coordinates(atoms, coord_file)
            
            # Run TBLite calculation
            energy, forces, stress = self._run_tblite_calculation(tmpdir, coord_file)
            
            # Set ASE results
            self.results['energy'] = energy
            if 'forces' in properties:
                self.results['forces'] = forces
            if 'stress' in properties:
                self.results['stress'] = stress
    
    def _write_coordinates(self, atoms: Atoms, coord_file: Path):
        """Write atomic coordinates to XYZ file"""
        with open(coord_file, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"TBLite calculation with {self.method}\n")
            for atom in atoms:
                f.write(f"{atom.symbol} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n")
    
    def _run_tblite_calculation(self, tmpdir: str, coord_file: Path) -> tuple:
        """Run TBLite calculation and parse results"""
        # Build TBLite command
        cmd = [
            "tblite", "run",
            "--method", self.method,
            "--param", str(self.param_file.resolve()),
            "--etemp", str(self.electronic_temperature),
            "--grad", "tblite.txt",  # Output gradient to file
            str(coord_file)
        ]
        
        # Add optional parameters
        if self.charge != 0.0:
            cmd.extend(["--charge", str(self.charge)])
        if self.spin > 0:
            cmd.extend(["--spin", str(self.spin)])
        
        # Run TBLite
        try:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse results
            energy = self._parse_energy(result.stdout)
            forces, stress = self._parse_gradient(Path(tmpdir) / "tblite.txt")
            
            return energy, forces, stress
            
        except subprocess.CalledProcessError as e:
            logger.error(f"TBLite calculation failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"TBLite calculation failed: {e}")
    
    def _parse_energy(self, stdout: str) -> float:
        """Parse energy from TBLite output"""
        # Look for "total energy" line in stdout
        for line in stdout.split('\n'):
            if 'total energy' in line.lower():
                # Extract the last number on the line (the energy value)
                return float(line.split()[-1])
        
        raise ValueError("Could not parse energy from TBLite output")
    
    def _parse_gradient(self, grad_file: Path) -> tuple:
        """Parse forces and stress from gradient file"""
        if not grad_file.exists():
            raise FileNotFoundError(f"Gradient file not found: {grad_file}")
        
        with open(grad_file, 'r') as f:
            content = f.read()
        
        # Parse forces (gradients)
        forces = []
        if 'gradient' in content:
            # Find gradient section and extract force vectors
            grad_start = content.find('gradient')
            grad_lines = content[grad_start:].split('\n')[1:]  # Skip header
            
            for line in grad_lines:
                if line.strip() and not line.startswith('virial'):
                    try:
                        fx, fy, fz = map(float, line.split()[:3])
                        forces.append([-fx, -fy, -fz])  # Convert gradient to force
                    except (ValueError, IndexError):
                        break
                elif line.startswith('virial'):
                    break
        
        # Parse stress tensor (virial)
        stress = None
        if 'virial' in content:
            # Find virial section and extract 3x3 matrix
            virial_start = content.find('virial')
            virial_lines = content[virial_start:].split('\n')[1:]  # Skip header
            
            stress_matrix = []
            for line in virial_lines[:3]:  # Take first 3 lines
                if line.strip():
                    try:
                        row = list(map(float, line.split()[:3]))
                        stress_matrix.append(row)
                    except (ValueError, IndexError):
                        break
            
            if len(stress_matrix) == 3:
                # Convert to ASE stress format (Voigt notation)
                stress = np.array([
                    stress_matrix[0][0], stress_matrix[1][1], stress_matrix[2][2],
                    stress_matrix[1][2], stress_matrix[0][2], stress_matrix[0][1]
                ])
        
        if not forces:
            raise ValueError("Could not parse forces from gradient file")
        
        return np.array(forces), stress
    
    def get_forces(self, atoms: Optional[Atoms] = None) -> np.ndarray:
        """Get forces on atoms"""
        if atoms is not None:
            self.calculate(atoms, properties=['forces'])
        return self.results.get('forces')
    
    def get_stress(self, atoms: Optional[Atoms] = None) -> np.ndarray:
        """Get stress tensor"""
        if atoms is not None:
            self.calculate(atoms, properties=['stress'])
        return self.results.get('stress') 
