import numpy as np
import re
import json
import os
import uuid
from ase.dft.bandgap import bandgap as ase_bandgap
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
        # Cache for ASE bandgap support
        self._last_eigenvalues_eV = None  # type: Optional[np.ndarray]
        self._last_fermi_eV = None  # type: Optional[float]
        
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
            energy, forces, stress = self._run_tblite_calculation(tmpdir, coord_file, atoms)
            
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
    
    def _run_tblite_calculation(self, tmpdir: str, coord_file: Path, atoms: Atoms) -> tuple:
        """Run TBLite calculation and parse results"""
        # Build TBLite command with enhanced convergence parameters
        cmd = [
            "tblite", "run",
            "--method", self.method,
            "--param", str(self.param_file.resolve()),
            "--iterations", "500",  # Increased iterations for better convergence
            "--etemp", str(self.electronic_temperature),
            "--grad", "tblite.txt",  # Output gradient to file
            "--json", "tblite.json",
            "-v",
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
        # Optional: echo stdout/stderr for debugging
        if os.environ.get('TBLITE_PRINT_STDOUT'):
            print("\n[TBLite stdout]\n" + (result.stdout or ""))
            print("\n[TBLite stderr]\n" + (result.stderr or ""))
        
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
                    "--etemp", str(self.electronic_temperature),
                    "--grad", "tblite.txt",
                    "--json", "tblite.json",
                    "-v",
                    str(coord_file)
                ]
                
                result_relaxed = subprocess.run(
                    cmd_relaxed,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=False)
                if os.environ.get('TBLITE_PRINT_STDOUT'):
                    print("\n[TBLite relaxed stdout]\n" + (result_relaxed.stdout or ""))
                    print("\n[TBLite relaxed stderr]\n" + (result_relaxed.stderr or ""))
                
                if result_relaxed.returncode == 0:
                    logger.info(f"Calculation succeeded with relaxed convergence")
                    energy = self._parse_energy(result_relaxed.stdout)
                    gap = self._parse_bandgap(result_relaxed.stdout)
                    # Prefer JSON-derived gap if available
                    json_gap = self._parse_json_bandgap(Path(tmpdir) / "tblite.json")
                    if json_gap is not None:
                        gap = json_gap
                    forces, stress = self._parse_gradient(Path(tmpdir) / "tblite.txt")
                    self._maybe_persist_json(Path(tmpdir) / "tblite.json")
                    # Fallback: try ASE Python API for bandgap if still missing
                    if gap is None:
                        try:
                            gap_ase = self._compute_gap_via_ase(atoms)
                            if gap_ase is not None:
                                gap = gap_ase
                        except Exception:
                            pass
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
        # Prefer JSON-derived gap if available
        json_gap = self._parse_json_bandgap(Path(tmpdir) / "tblite.json")
        if json_gap is not None:
            gap = json_gap
        forces, stress = self._parse_gradient(Path(tmpdir) / "tblite.txt")
        self._maybe_persist_json(Path(tmpdir) / "tblite.json")
        # Fallback: try ASE Python API for bandgap if still missing
        if gap is None:
            try:
                gap_ase = self._compute_gap_via_ase(atoms)
                if gap_ase is not None:
                    gap = gap_ase
            except Exception:
                pass
        if gap is not None:
            self.results['bandgap'] = gap
        
        return energy, forces, stress

    def _parse_bandgap(self, output: str) -> Optional[float]:
        """Parse HOMO-LUMO gap from TBLite stdout.
        
        TBLite outputs a clear "HL-Gap" line with the gap in eV.
        Example: "HL-Gap            0.0031795 Eh            0.0865 eV"
        """
        for line in output.splitlines():
            if 'HL-Gap' in line or 'hl-gap' in line.lower():
                # Look for the eV value in the line
                tokens = line.split()
                for i, token in enumerate(tokens):
                    if token.lower() == 'ev':
                        # The previous token should be the gap value
                        try:
                            return float(tokens[i-1])
                        except (ValueError, IndexError):
                            continue
                # Fallback: look for any float followed by 'eV' anywhere in the line
                import re
                ev_match = re.search(r'(\d+\.?\d*)\s*eV', line, re.IGNORECASE)
                if ev_match:
                    return float(ev_match.group(1))
        
        return None

    def _maybe_persist_json(self, json_path: Path) -> None:
        """Optionally persist tblite.json to a user-specified directory via
        environment variable TBLITE_SAVE_JSON_DIR for debugging/inspection.
        """
        try:
            save_dir = os.environ.get('TBLITE_SAVE_JSON_DIR')
            if not save_dir:
                return
            if not json_path.exists():
                return
            dst_dir = Path(save_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            unique = uuid.uuid4().hex[:8]
            dst = dst_dir / f"tblite_{self.method}_{unique}.json"
            with open(json_path, 'r') as src_f, open(dst, 'w') as dst_f:
                dst_f.write(src_f.read())
        except Exception:
            # Best-effort only
            pass

    def _parse_json_bandgap(self, json_file: Path) -> Optional[float]:
        """Parse HOMO-LUMO gap from TBLite JSON results.
        
        TBLite JSON may contain explicit gap values or orbital energies.
        Returns gap in eV when possible.
        """
        try:
            if not json_file.exists():
                return None
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception:
            return None

        # Look for explicit gap value in JSON (similar to stdout parsing)
        def find_gap_value(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if 'gap' in kl and isinstance(v, (int, float)):
                        return float(v)
                    result = find_gap_value(v)
                    if result is not None:
                        return result
            elif isinstance(node, list):
                for item in node:
                    result = find_gap_value(item)
                    if result is not None:
                        return result
            return None

        gap = find_gap_value(data)
        if gap is not None:
            return float(gap)

        return None

    # --- Minimal ASE calculator API for bandgap(calc) convenience ---
    def get_eigenvalues(self, kpt: Optional[int] = None, spin: Optional[int] = None) -> np.ndarray:
        """Return eigenvalue array in eV for the last calculation.
        
        Args:
            kpt: k-point index (default: 0)
            spin: spin index (default: 0)
            
        Returns:
            1D array of eigenvalues in eV
        """
        if self._last_eigenvalues_eV is None:
            raise RuntimeError("No eigenvalues cached. Run a calculation first (with --json enabled).")
        
        arr = self._last_eigenvalues_eV
        
        # Handle different array shapes
        if arr.ndim == 3:
            # Shape (nspin, nkpt, nband)
            s_idx = spin if spin is not None else 0
            k_idx = kpt if kpt is not None else 0
            return arr[s_idx, k_idx, :]
        elif arr.ndim == 2:
            # Shape (nkpt, nband) - single spin
            k_idx = kpt if kpt is not None else 0
            return arr[k_idx, :]
        else:
            # Shape (nband,) - single k-point, single spin
            return arr.reshape(-1)

    def get_fermi_level(self) -> Optional[float]:
        """Return Fermi level in eV (if available)."""
        return self._last_fermi_eV

    def get_number_of_spins(self) -> int:
        """Return number of spin channels.
        
        For TBLite, this is determined by the spin parameter:
        - spin = 0: closed shell (1 spin channel)
        - spin > 0: open shell (2 spin channels)
        """
        return 2 if self.spin > 0 else 1

    def get_ibz_k_points(self) -> np.ndarray:
        """Return Gamma-only k-point to satisfy ASE bandgap API when needed."""
        return np.zeros((1, 3), dtype=float)
    
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
