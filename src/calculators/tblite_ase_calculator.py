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
    
    implemented_properties = ['energy', 'forces', 'stress', 'bandgap']
    
    def __init__(self, 
                 param_file: str,
                 method: str = "gfn1",
                 electronic_temperature: float = 500.0,
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
            # Write coordinates to file in $coord format expected by tblite
            coord_file = Path(tmpdir) / "coord"
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
        """Write atomic coordinates to 'coord' file with optional periodic cell.

        TBLite understands a Turbomole-like $coord format. We write positions in Bohr.
        If periodic, also include $cell and $periodic 3 sections.
        """
        with open(coord_file, 'w') as f:
            f.write("$coord\n")
            if isinstance(atoms, Atoms):
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                cell = atoms.get_cell()
                pbc = any(atoms.get_pbc())
            else:
                symbols = [a.symbol for a in atoms]
                positions = [a.position for a in atoms]
                cell = None
                pbc = False

            # Write atomic positions (Angstrom -> Bohr)
            for symbol, pos in zip(symbols, positions):
                x_bohr = pos[0] * 1.88973
                y_bohr = pos[1] * 1.88973
                z_bohr = pos[2] * 1.88973
                f.write(f"{x_bohr:.10f} {y_bohr:.10f} {z_bohr:.10f} {symbol.lower()}\n")
            f.write("$end\n")

            if pbc and cell is not None and cell.any():
                f.write("$cell\n")
                for i in range(3):
                    f.write(f"{cell[i, 0]:.10f} {cell[i, 1]:.10f} {cell[i, 2]:.10f}\n")
                f.write("$end\n")
                f.write("$periodic 3\n")
                f.write("$end\n")
    
    def _run_tblite_calculation(self, tmpdir: str, coord_file: Path) -> tuple:
        """Run TBLite calculation and parse results"""
        # Build TBLite command with enhanced convergence parameters
        cmd = [
            "tblite", "run",
            "--method", self.method,
            "--param", str(self.param_file.resolve()),
            "--iterations", "500",  # Increased iterations for better convergence
            "--etemp", str(self.electronic_temperature + 150),
            "--grad", "tblite.txt",  # Output gradient to file
            str(coord_file)
        ]
        
        # Add optional parameters
        if self.charge != 0.0:
            cmd.extend(["--charge", str(self.charge)])
        if self.spin > 0:
            cmd.extend(["--spin", str(self.spin)])
        
        # Run TBLite with convergence handling
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            # Check if it's a convergence issue
            stderr_lower = result.stderr.lower()
            if "convergence" in stderr_lower or "scf" in stderr_lower or "iterations" in stderr_lower:
                # Try with more relaxed convergence
                logger.warning(f"SCF convergence failed, trying with relaxed parameters...")
                cmd_relaxed = [
                    "tblite", "run",
                    "--method", self.method,
                    "--param", str(self.param_file.resolve()),
                    "--iterations", "1000",  # More iterations
                    "--etemp", str(self.electronic_temperature + 150),
                    "--grad", "tblite.txt",
                    str(coord_file)
                ]
                
                result_relaxed = subprocess.run(
                    cmd_relaxed,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=False)
                
                if result_relaxed.returncode == 0:
                    logger.info(f"Calculation succeeded with relaxed convergence")
                    energy = self._parse_energy(result_relaxed.stdout)
                    gap = self._parse_bandgap(result_relaxed.stdout)
                    forces, stress = self._parse_gradient(Path(tmpdir) / "tblite.txt")
                    if gap is not None:
                        self.results['bandgap'] = gap
                    return energy, forces, stress
                else:
                    
            
                    # If all attempts failed, raise error with detailed information
                    error_msg = f"TBLite calculation failed with exit code {result.returncode}\n"
                    error_msg += f"Command: {' '.join(cmd)}\n"
                    error_msg += f"Working directory: {tmpdir}\n"
                    error_msg += f"Parameter file: {self.param_file}\n"
                    error_msg += f"STDOUT:\n{result.stdout}\n"
                    error_msg += f"STDERR:\n{result.stderr}"
                    raise RuntimeError(error_msg)
        
        # Parse results from successful calculation
        energy = self._parse_energy(result.stdout)
        gap = self._parse_bandgap(result.stdout)
        forces, stress = self._parse_gradient(Path(tmpdir) / "tblite.txt")
        if gap is not None:
            self.results['bandgap'] = gap
        
        return energy, forces, stress

    def _parse_bandgap(self, output: str) -> Optional[float]:
        """Parse HOMO-LUMO/band gap from TBLite stdout if available.
        Returns gap in eV when possible.
        """
        text = output.lower()
        # Try common patterns
        candidates = []
        for line in output.splitlines():
            lower = line.lower()
            if 'homo' in lower and 'lumo' in lower and 'gap' in lower:
                candidates.append(line)
            elif 'band gap' in lower:
                candidates.append(line)
        for line in candidates:
            tokens = line.replace('=', ' ').replace(':', ' ').split()
            # Look for a float followed by eV or a float alone
            for i, tok in enumerate(tokens):
                try:
                    val = float(tok)
                    # If unit token next
                    unit = tokens[i+1].lower() if i + 1 < len(tokens) else ''
                    if 'ev' in unit:
                        return val
                    if 'eh' in unit:
                        return val * 27.211386245988  # Eh -> eV
                    # If no unit, assume eV
                    return val
                except Exception:
                    continue
        return None
    
    def _parse_energy(self, output):
        # DEBUG: Extract energy from the summary 'total energy' line (not the cycle table)
        for line in output.splitlines():
            if 'total energy' in line.lower():
                tokens = line.split()
                # Find the first token after 'total energy' that can be parsed as a float
                for i, token in enumerate(tokens):
                    if token.lower() == 'energy':
                        # Look for the next token that can be parsed as a float
                        for next_token in tokens[i+1:]:
                            try:
                                return float(next_token)
                            except ValueError:
                                continue
                # Fallback: try the last token
                try:
                    return float(tokens[-2])  # -2 because last is 'Eh'
                except Exception:
                    pass
        # DEBUG: If no valid energy found, log and raise
        print(f"[DEBUG] No valid summary 'total energy' found in output:\n{output}")
        raise ValueError("[DEBUG] No valid summary 'total energy' found in output.")
    
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
