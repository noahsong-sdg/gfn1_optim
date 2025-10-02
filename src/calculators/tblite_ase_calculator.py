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
            "--etemp", str(self.electronic_temperature + 150),
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
                    "--etemp", str(self.electronic_temperature + 150),
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
        """Parse HOMO-LUMO/band gap from TBLite stdout if available.
        Returns gap in eV when possible.
        """
        text = output.lower()
        # Try common patterns where the gap is reported explicitly
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
        
        # Fallback: infer from separate HOMO and LUMO energies if present
        homo_val: Optional[float] = None
        lumo_val: Optional[float] = None
        homo_unit: Optional[str] = None
        lumo_unit: Optional[str] = None
        # Capture lines where label may appear after the numeric value as well
        # e.g., "-10.5927  (HOMO)" or "-7.2301  ( LUMO )"
        homo_pattern = re.compile(r"(?:homo[^\n]*?([\-+]?\d+(?:\.\d+)?)(?:\s*(ev|eh))?)|([\-+]?\d+(?:\.\d+)?)\s*\(\s*homo\s*\)", re.IGNORECASE)
        lumo_pattern = re.compile(r"(?:lumo[^\n]*?([\-+]?\d+(?:\.\d+)?)(?:\s*(ev|eh))?)|([\-+]?\d+(?:\.\d+)?)\s*\(\s*lumo\s*\)", re.IGNORECASE)
        for line in output.splitlines():
            try:
                if homo_val is None and ('homo' in line.lower()):
                    m = homo_pattern.search(line)
                    if m:
                        # Prefer named-capture group order: pre-labeled or post-labeled number
                        num = m.group(1) or m.group(3)
                        unit = m.group(2)
                        homo_val = float(num)
                        homo_unit = (unit or 'eV').lower()
                if lumo_val is None and ('lumo' in line.lower()):
                    m = lumo_pattern.search(line)
                    if m:
                        num = m.group(1) or m.group(3)
                        unit = m.group(2)
                        lumo_val = float(num)
                        lumo_unit = (unit or 'eV').lower()
            except Exception:
                continue
        def to_ev(value: float, unit: Optional[str]) -> float:
            if unit is None:
                return value
            if 'eh' in unit:
                return value * 27.211386245988
            return value  # assume eV
        if homo_val is not None and lumo_val is not None:
            homo_ev = to_ev(homo_val, homo_unit)
            lumo_ev = to_ev(lumo_val, lumo_unit)
            gap = lumo_ev - homo_ev
            # Ensure non-negative gap
            if gap < 0:
                gap = abs(gap)
            return float(gap)
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
        """Parse bandgap from TBLite JSON results if present. Falls back to
        computing HOMO-LUMO gap from eigenvalues and occupations.
        Returns gap in eV when possible.
        """
        try:
            if not json_file.exists():
                return None
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception:
            return None

        # Try ASE's bandgap() first if we can extract eigenvalues (+ optional Fermi)
        def find_first_array(node, key_candidates):
            if isinstance(node, dict):
                for k, v in node.items():
                    if any(c in str(k).lower() for c in key_candidates) and isinstance(v, list):
                        return v
                    found = find_first_array(v, key_candidates)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = find_first_array(item, key_candidates)
                    if found is not None:
                        return found
            return None

        eigenvalues = find_first_array(data, ['eigenvalues', 'orbital-energies', 'eigen', 'orbitals'])
        def flatten_if_nested(arr):
            if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
                flat = []
                for sub in arr:
                    if isinstance(sub, list):
                        flat.extend(sub)
                return flat
            return arr

        def to_ev_array(raw):
            try:
                arr = np.array([float(x) for x in raw], dtype=float)
            except Exception:
                return None
            # Heuristic: Hartree -> eV if |E|max < 5
            if np.nanmax(np.abs(arr)) < 5.0:
                arr = arr * 27.211386245988
            return arr

        if eigenvalues is not None:
            eigenvalues = flatten_if_nested(eigenvalues)
            eig_arr = to_ev_array(eigenvalues)
            if eig_arr is not None and eig_arr.size > 0:
                # Shape to (nspin, nkpt, nband)
                nband = eig_arr.size
                eig_reshaped = eig_arr.reshape((1, 1, nband))
                # Optional Fermi
                def find_numeric(node, keys):
                    if isinstance(node, dict):
                        for k, v in node.items():
                            if any(c in str(k).lower() for c in keys) and isinstance(v, (int, float)):
                                return float(v)
                            found = find_numeric(v, keys)
                            if found is not None:
                                return found
                    elif isinstance(node, list):
                        for item in node:
                            found = find_numeric(item, keys)
                            if found is not None:
                                return found
                    return None
                fermi = find_numeric(data, ['fermi', 'e_fermi', 'fermi_level'])
                if fermi is not None and abs(fermi) < 5.0:
                    fermi = fermi * 27.211386245988
                try:
                    gap_val, _, _ = ase_bandgap(eigenvalues=eig_reshaped, efermi=fermi)
                    # Cache for ASE interface methods
                    self._last_eigenvalues_eV = eig_reshaped
                    self._last_fermi_eV = fermi
                    if gap_val is not None and np.isfinite(gap_val):
                        return float(abs(gap_val))
                except Exception:
                    # Still cache even if bandgap calc fails
                    self._last_eigenvalues_eV = eig_reshaped
                    self._last_fermi_eV = fermi
                    pass

        # 1) Try to find an explicit numeric gap anywhere in the JSON
        def find_numeric_gap(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if 'gap' in kl and isinstance(v, (int, float)):
                        return float(v)
                    found = find_numeric_gap(v)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = find_numeric_gap(item)
                    if found is not None:
                        return found
            return None

        explicit_gap = find_numeric_gap(data)
        if explicit_gap is not None:
            return float(explicit_gap)

        # 2) Try to compute from eigenvalues and occupations
        def find_first_array(node, key_candidates):
            if isinstance(node, dict):
                for k, v in node.items():
                    if any(c in str(k).lower() for c in key_candidates) and isinstance(v, list):
                        return v
                    found = find_first_array(v, key_candidates)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = find_first_array(item, key_candidates)
                    if found is not None:
                        return found
            return None

        eigenvalues = find_first_array(data, ['orbital', 'eigen', 'eigenvalues'])
        occupations = find_first_array(data, ['occupation', 'occupations', 'occ'])

        # Flatten possible spin-channels [[...], [...]]
        def flatten_if_nested(arr):
            if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
                flat = []
                for sub in arr:
                    if isinstance(sub, list):
                        flat.extend(sub)
                return flat
            return arr

        if eigenvalues is None:
            return None
        eigenvalues = flatten_if_nested(eigenvalues)
        if occupations is not None:
            occupations = flatten_if_nested(occupations)

        try:
            ev = np.array([float(x) for x in eigenvalues], dtype=float)
        except Exception:
            return None

        # Heuristic unit detection: if orbital energies look like Hartree values, convert to eV
        def convert_to_ev(values: np.ndarray) -> np.ndarray:
            # If absolute values are small (|E| < 5), assume Hartree and convert
            if np.nanmax(np.abs(values)) < 5.0:
                return values * 27.211386245988
            return values

        ev_eV = convert_to_ev(ev)

        # Use occupations if available; otherwise try Fermi level
        if occupations is not None:
            try:
                occ = np.array([float(x) for x in occupations], dtype=float)
                # Consider occupied if occupancy > ~0
                occ_mask = occ > 1e-6
            except Exception:
                occ_mask = None
        else:
            occ_mask = None

        if occ_mask is not None and occ_mask.shape == ev_eV.shape:
            if not np.any(occ_mask) or np.all(occ_mask):
                return None
            homo = np.max(ev_eV[occ_mask])
            lumo = np.min(ev_eV[~occ_mask])
            gap = float(lumo - homo)
            if gap < 0:
                gap = abs(gap)
            return gap

        # Try Fermi level key if occupations missing
        def find_numeric(node, keys):
            if isinstance(node, dict):
                for k, v in node.items():
                    if any(c in str(k).lower() for c in keys) and isinstance(v, (int, float)):
                        return float(v)
                    found = find_numeric(v, keys)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = find_numeric(item, keys)
                    if found is not None:
                        return found
            return None

        fermi = find_numeric(data, ['fermi', 'e_fermi', 'fermi_level'])
        if fermi is not None:
            # Convert Fermi to eV if needed (same heuristic as eigenvalues)
            fermi_eV = fermi * 27.211386245988 if abs(fermi) < 5.0 else fermi
            occ_mask = ev_eV <= fermi_eV + 1e-6
            if not np.any(occ_mask) or np.all(occ_mask):
                return None
            homo = np.max(ev_eV[occ_mask])
            lumo = np.min(ev_eV[~occ_mask])
            gap = float(lumo - homo)
            if gap < 0:
                gap = abs(gap)
            # Cache
            self._last_eigenvalues_eV = ev_eV.reshape((1, 1, ev_eV.size))
            self._last_fermi_eV = fermi_eV
            return gap

        return None

    # --- Minimal ASE calculator API for bandgap(calc) convenience ---
    def get_eigenvalues(self, kpt: Optional[int] = None, spin: Optional[int] = None) -> np.ndarray:
        """Return 1D eigenvalue array in eV for the last calculation.
        Ignores kpt/spin and returns Gamma-only flattened spectrum if present.
        """
        if self._last_eigenvalues_eV is None:
            raise RuntimeError("No eigenvalues cached. Run a calculation first (with --json enabled).")
        arr = self._last_eigenvalues_eV
        if arr.ndim == 3:
            return arr[0, 0, :]
        return arr.reshape(-1)

    def get_fermi_level(self) -> Optional[float]:
        """Return Fermi level in eV (if available)."""
        return self._last_fermi_eV

    def get_ibz_k_points(self) -> np.ndarray:
        """Return Gamma-only k-point to satisfy ASE bandgap API when needed."""
        return np.zeros((1, 3), dtype=float)
    
    def _compute_gap_via_ase(self, atoms: Atoms) -> Optional[float]:
        """Attempt to compute bandgap using the tblite.ase Python API and ASE's
        bandgap utility. This is a fallback when CLI/stdout/JSON do not provide
        the gap directly.
        """
        try:
            from tblite.ase import TBLite as TBLiteASE
        except Exception:
            return None
        try:
            # Prefer providing method and param; charge/spin may not always be supported
            ase_calc = TBLiteASE(method=self.method, param=str(self.param_file.resolve()))
        except Exception:
            try:
                ase_calc = TBLiteASE(method=self.method)
            except Exception:
                return None
        try:
            atoms_local = atoms.copy()
            atoms_local.calc = ase_calc
            _ = atoms_local.get_potential_energy()
            gap_val, _, _ = ase_bandgap(ase_calc)
            if gap_val is not None and np.isfinite(gap_val):
                return float(abs(gap_val))
        except Exception:
            pass
        try:
            ev = ase_calc.get_eigenvalues(kpt=0)
            if ev is None or len(ev) == 0:
                return None
            ev = np.array(ev, dtype=float)
            try:
                fermi = ase_calc.get_fermi_level()
            except Exception:
                fermi = None
            if fermi is None:
                return None
            homo = np.max(ev[ev <= fermi + 1e-6])
            above = ev[ev > fermi + 1e-6]
            if above.size == 0:
                return None
            lumo = np.min(above)
            return float(abs(lumo - homo))
        except Exception:
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
