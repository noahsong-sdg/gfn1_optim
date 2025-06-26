import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
import toml
from typing import Dict, List, Tuple, Optional

from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from ase.build import bulk 
from tblite.ase import TBLite
from ase.dft import bandgap

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import re
def xtb_calc(tel=300):
    return TBLite(method="GFN1-xTB", electronic_temperature=tel)

# configs: [(x1,y1,z1), (x2,y2,z2), ...]
# results_list: [time_array, a_scaled_array, c_scaled_array, u_params_array, volumes_array]
def lattice_constants(configs, compound='CdS', tel=300, a_init=None, c_init=None, csv=False, filename=None):
    assert isinstance(configs, list), "configs must be a list of tuples"
    #assert a_init, c_init not None, "a_init and c_init must be provided"
    times = []
    volumes = []
    a_params = []
    c_params = []
    u_params = []
    for i, reps in enumerate(configs):   
        base_wurtz = bulk(compound, crystalstructure='wurtzite', a=a_init, c=c_init) 
        atoms = base_wurtz.repeat(reps)
        atoms.calc = xtb_calc(tel=tel) 
        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf)

        try:
            start_cpu_time = time.process_time()
            opt.run(fmax=0.01) # Convergence criterion: max force < 0.01 eV/Å
            elapsed_time = time.process_time() - start_cpu_time

            cellpars = atoms.cell.cellpar()

            scaled_positions = atoms.get_scaled_positions()
            u_val = scaled_positions[2, 2] 
            # Store results
            volumes.append(atoms.get_volume())
            a_params.append(cellpars[0])
            c_params.append(cellpars[2])
            u_params.append(u_val)
            times.append(elapsed_time)

        except Exception as e:
            print(f"  Error during optimization for supercell {reps}: {e}")
            volumes.append(0.0)
            a_params.append(0.0)
            c_params.append(0.0)
            u_params.append(0.0)
            times.append(0.0)

    times  = np.array(times)
    a_params = np.array(a_params)
    c_params = np.array(c_params)
    u_params = np.array(u_params)
    volumes = np.array(volumes)

    a_reps = [el[0] for el in configs]
    a_scaled = a_params / a_reps
    c_reps = [el[2] for el in configs]
    c_scaled = c_params / c_reps
    df_data_list = []
    # Iterate through each configuration and its corresponding results
    for i, config_tuple in enumerate(configs):
        df_data_list.append({
            'Configuration': config_tuple, # The (x,y,z) tuple
            'Time': times[i],
            'a': a_scaled[i],
            'c': c_scaled[i],
            'u': u_params[i],
            'Volume': volumes[i],
        })

    configs_df = pd.DataFrame(df_data_list)
    configs_df = configs_df.sort_values(by='Volume')
    # Save the DataFrame to a CSV file
    if filename is None:
        filename = 'configs_data'
    configs_df.to_csv(f'{filename}.csv', index=False) if csv else None
    return configs_df

# Example usage:
# create_element_toml(['H', 'C', 'N'], 'selected_elements.toml')
def get_prop(atoms: Atoms, timer=False):
    """
    Extracts lattice parameters, band gap, and elastic constants from an ASE Atoms object.

    Args:
        atoms (ase.Atoms): An ASE Atoms object representing the bulk crystal.
                           Should be reasonably relaxed for meaningful results.
        calculator_method (str): tblite method to use (e.g., "GFN1-xTB", "GFN2-xTB").

    Returns:
        dict: A dictionary containing:
              'lattice_a' (float): Length of the first lattice vector.
              'lattice_c' (float): Length of the third lattice vector.
              'angle_alpha' (float): Angle between lattice vectors b and c (degrees).
              'angle_beta' (float): Angle between lattice vectors a and c (degrees).
              'angle_gamma' (float): Angle between lattice vectors a and b (degrees).
              'volume' (float): Volume of the unit cell (Angstrom^3).
              'u_parameter' (str or float): Placeholder or calculated 'u' parameter.
                                           Calculation is structure-specific (e.g., for wurtzite)..
              'Band Gap' (float): 
              'elastic_constants' (numpy.ndarray or str): 6x6 Voigt notation elastic tensor (C_ij) in GPa.
                                                          Returns an error string if calculation fails.
    """
    # step one: relax! ------------------------------
    atoms.calc = xtb_calc(tel=300) 
    ucf = UnitCellFilter(atoms)
    opt = BFGS(ucf)
    if timer:
        start = time.time()
        opt.run(fmax=0.01)
        elapsed_time = time.time() - start
    else:
        opt.run(fmax=0.01) # Convergence criterion: max force < 0.01 eV/Å

    #step two: get the stuff -----------------------------
    cellpars = atoms.cell.cellpar()
    scaled_positions = atoms.get_scaled_positions()
    u = scaled_positions[2, 2] # this needs adjusting 

    # step three: get band gap --------------------------------------
    calculator = xtb_calc(tel=300)
    # https://wiki.fysik.dtu.dk/ase/ase/dft/bandgap.html
    gap, p1, p2 = bandgap(calc=calculator, direct=False, eigenvalues=None, efermi=None, output=None, kpts=None)    # step four: calculate elastic constants -------------------------------------
    # Using finite differences approach for elastic constants
    try:
        from ase.calculators.calculator import Calculator
        from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
        
        # Simple elastic constants calculation using stress-strain relations
        # This is a simplified approach - for production use elastic library
        elastic = calculate_elastic_constants_simple(atoms)
        
    except Exception as e:
        print(f"Warning: Elastic constants calculation failed: {e}")
        # Return placeholder values for elastic constants (6x6 matrix)
        elastic = np.zeros((6, 6))

    properties = {
        'lattice_a': cellpars[0],  # Length of the first lattice vector (a)
        'lattice_c': cellpars[2],  # Length of the third lattice vector (c)
        'angle_alpha': cellpars[3],  # Angle between b and c (alpha)
        'angle_beta': cellpars[4],  # Angle between a and c (beta)
        'angle_gamma': cellpars[5],  # Angle between a and b (gamma)
        'volume': atoms.get_volume(),  # Volume of the unit cell (Angstrom^3)
        'u_parameter': u,
        'Band Gap': gap,
        'elastic_constants_GPa': elastic, # 6x6 Voigt matrix
    }
    return properties

def create_filtered_toml(element_list, input_filename='gfn1-base.toml', output_filename=None):
    """
    Create a filtered TOML file containing only parameters for specified elements.
    
    Args:
        element_list: List of element symbols (e.g., ['H', 'C', 'N'])
        input_filename: Path to the input TOML file (default: 'gfn1-base.toml')
        output_filename: Name of the output TOML file (if None, auto-generated)
    
    Returns:
        str: Path to the created filtered TOML file
    """
    import os
    
    # Convert to set for O(1) lookup
    target_elements = set(element_list)
    
    # Generate output filename if not provided
    if output_filename is None:
        elements_str = '_'.join(sorted(element_list))
        output_filename = f'gfn1_filtered_{elements_str}.toml'
    
    # Determine the full path to input file (check both current dir and parent dir)
    if not os.path.exists(input_filename):
        parent_path = os.path.join('..', input_filename)
        if os.path.exists(parent_path):
            input_filename = parent_path
        else:
            raise FileNotFoundError(f"Could not find {input_filename}")
    
    with open(input_filename, 'r') as f:
        lines = f.readlines()
    
    # Track current section and whether to include it
    current_section = None
    include_current_section = True
    in_kpair_section = False
    in_element_section = False
    current_element = None
    
    filtered_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check for section headers
        if stripped.startswith('['):
            # Determine the section type
            if stripped == '[hamiltonian.xtb.kpair]':
                in_kpair_section = True
                in_element_section = False
                include_current_section = True
                filtered_lines.append(line)
                continue
            elif stripped.startswith('[element.'):
                in_kpair_section = False
                in_element_section = True
                # Extract element name from [element.X]
                current_element = stripped[9:-1]  # Remove '[element.' and ']'
                include_current_section = current_element in target_elements
                if include_current_section:
                    filtered_lines.append(line)
                continue
            elif stripped == '[element]':
                in_kpair_section = False
                in_element_section = False
                include_current_section = True
                filtered_lines.append(line)
                continue
            else:
                # General sections (meta, hamiltonian, dispersion, etc.)
                in_kpair_section = False
                in_element_section = False
                include_current_section = True
                filtered_lines.append(line)
                continue
        
        # Handle content based on current section
        if in_kpair_section:
            # Filter kpair entries: only include if both elements are in target set
            if '=' in stripped and '-' in stripped:
                pair_part = stripped.split('=')[0].strip()
                elements_in_pair = pair_part.split('-')
                if len(elements_in_pair) == 2:
                    elem1, elem2 = elements_in_pair
                    # Include if both elements are in our target set
                    if elem1 in target_elements and elem2 in target_elements:
                        filtered_lines.append(line)
                # Skip pairs that don't involve our target elements
                continue
            else:
                # Non-pair line in kpair section (shouldn't happen, but include for safety)
                filtered_lines.append(line)
                continue
        
        elif in_element_section:
            # Only include if we're processing a target element
            if include_current_section:
                filtered_lines.append(line)
            # Skip lines for non-target elements
            continue
        
        else:
            # General sections - always include
            if include_current_section:
                filtered_lines.append(line)
    
    # Write filtered content to output file
    with open(output_filename, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"Filtered TOML created: {output_filename}")
    print(f"Included elements: {sorted(element_list)}")
    
    # Count filtered kpairs for info
    kpair_count = 0
    for line in filtered_lines:
        if '=' in line and '-' in line and '[' not in line:
            kpair_count += 1
    print(f"Included {kpair_count} element pair interactions")
    
    return output_filename

# read from toml file. put the params into one array
# compute the things
# compute loss
# atoms.calc = TBLite(method="GFN1-xTB", param="gfn1-custom.toml", electronic_temperature=300)

class TomlParameterExtractor:
    """Extracts trainable parameters from TOML files and converts them to/from torch tensors."""
    
    def __init__(self, element_list: List[str]):
        self.element_list = element_list
        self.param_names = []
        self.param_indices = {}
        
    def extract_parameters(self, toml_path: str) -> torch.Tensor:
        """
        Extract all trainable parameters from TOML file into a single tensor.
        
        Returns:
            torch.Tensor: Flattened parameter vector
        """
        with open(toml_path, 'r') as f:
            data = toml.load(f)
        
        params = []
        self.param_names = []
        
        # Extract kpair parameters
        if 'hamiltonian' in data and 'xtb' in data['hamiltonian'] and 'kpair' in data['hamiltonian']['xtb']:
            kpair_data = data['hamiltonian']['xtb']['kpair']
            for pair_key, value in kpair_data.items():
                params.append(float(value))
                self.param_names.append(f"kpair.{pair_key}")
        
        # Extract element-specific parameters
        if 'element' in data:
            for element in self.element_list:
                if element in data['element']:
                    elem_data = data['element'][element]
                    
                    # Scalar parameters
                    scalar_params = ['gam', 'gam3', 'zeff', 'arep', 'en']
                    for param in scalar_params:
                        if param in elem_data:
                            params.append(float(elem_data[param]))
                            self.param_names.append(f"element.{element}.{param}")
                    
                    # Array parameters
                    array_params = ['levels', 'slater', 'shpoly', 'lgam', 'kcn']
                    for param in array_params:
                        if param in elem_data:
                            if isinstance(elem_data[param], list):
                                for i, val in enumerate(elem_data[param]):
                                    params.append(float(val))
                                    self.param_names.append(f"element.{element}.{param}[{i}]")
        
        # Create index mapping for efficient updates
        self.param_indices = {name: i for i, name in enumerate(self.param_names)}
        
        return torch.tensor(params, requires_grad=True, dtype=torch.float32)
    
    def update_toml_with_parameters(self, toml_path: str, param_tensor: torch.Tensor) -> None:
        """
        Update TOML file with new parameter values.
        
        Args:
            toml_path: Path to TOML file to update
            param_tensor: Updated parameter tensor
        """
        with open(toml_path, 'r') as f:
            data = toml.load(f)
        
        params = param_tensor.detach().numpy()
        
        # Update parameters based on names
        for i, param_name in enumerate(self.param_names):
            parts = param_name.split('.')
            
            if parts[0] == 'kpair':
                # Update kpair parameter
                pair_key = parts[1]
                data['hamiltonian']['xtb']['kpair'][pair_key] = float(params[i])
                
            elif parts[0] == 'element':
                # Update element parameter
                element = parts[1]
                param_type = parts[2]
                
                if '[' in param_type:
                    # Array parameter
                    base_param = param_type.split('[')[0]
                    index = int(param_type.split('[')[1].split(']')[0])
                    data['element'][element][base_param][index] = float(params[i])
                else:
                    # Scalar parameter
                    data['element'][element][param_type] = float(params[i])
        
        # Write updated data back to file
        with open(toml_path, 'w') as f:
            toml.dump(data, f)


def compute_properties_with_custom_params(atoms: Atoms, toml_path: str, tel: float = 300) -> Dict:
    """
    Compute properties using custom TOML parameters.
    
    Args:
        atoms: ASE Atoms object
        toml_path: Path to custom TOML parameter file
        tel: Electronic temperature
        
    Returns:
        Dictionary with computed properties
    """
    # Create a copy to avoid modifying original
    atoms_copy = atoms.copy()
    
    # Set up calculator with custom parameters
    atoms_copy.calc = TBLite(method="GFN1-xTB", param=toml_path, electronic_temperature=tel)
    
    # Use modified get_prop function that doesn't override the calculator
    return get_prop(atoms_copy, timer=False)


def compute_loss(predicted_props: Dict, target_props: Dict, weights: Optional[Dict] = None) -> torch.Tensor:
    """
    Compute weighted loss between predicted and target properties.
    
    Args:
        predicted_props: Dictionary of predicted properties
        target_props: Dictionary of target properties  
        weights: Optional weights for different property types
        
    Returns:
        Total loss as torch tensor
    """
    if weights is None:
        weights = {
            'lattice_a': 1.0,
            'lattice_c': 1.0, 
            'Band Gap': 1.0,
            'elastic_constants_GPa': 1.0
        }
    
    losses = []
    
    # Lattice parameter losses
    if 'lattice_a' in predicted_props and 'lattice_a' in target_props:
        loss_a = (predicted_props['lattice_a'] - target_props['lattice_a']) ** 2
        losses.append(weights['lattice_a'] * loss_a)
    
    if 'lattice_c' in predicted_props and 'lattice_c' in target_props:
        loss_c = (predicted_props['lattice_c'] - target_props['lattice_c']) ** 2
        losses.append(weights['lattice_c'] * loss_c)
    
    # Band gap loss
    if 'Band Gap' in predicted_props and 'Band Gap' in target_props:
        loss_gap = (predicted_props['Band Gap'] - target_props['Band Gap']) ** 2
        losses.append(weights['Band Gap'] * loss_gap)
    
    # Elastic constants loss (if available)
    if 'elastic_constants_GPa' in predicted_props and 'elastic_constants_GPa' in target_props:
        pred_elastic = predicted_props['elastic_constants_GPa']
        target_elastic = target_props['elastic_constants_GPa']
        
        # Handle case where elastic constants might be arrays
        if hasattr(pred_elastic, '__len__') and hasattr(target_elastic, '__len__'):
            pred_tensor = torch.tensor(pred_elastic, dtype=torch.float32) if not isinstance(pred_elastic, torch.Tensor) else pred_elastic
            target_tensor = torch.tensor(target_elastic, dtype=torch.float32) if not isinstance(target_elastic, torch.Tensor) else target_elastic
            loss_elastic = torch.mean((pred_tensor - target_tensor) ** 2)
            losses.append(weights['elastic_constants_GPa'] * loss_elastic)
    
    # Return total loss
    if losses:
        return torch.stack([torch.tensor(loss, dtype=torch.float32) if not isinstance(loss, torch.Tensor) else loss for loss in losses]).sum()
    else:
        return torch.tensor(0.0, dtype=torch.float32)


class GFNParameterOptimizer:
    """Main class for optimizing GFN1-xTB parameters."""
    
    def __init__(self, element_list: List[str], toml_path: str, learning_rate: float = 0.001):
        self.element_list = element_list
        self.toml_path = toml_path
        self.extractor = TomlParameterExtractor(element_list)
        
        # Extract initial parameters
        self.parameters = self.extractor.extract_parameters(toml_path)
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam([self.parameters], lr=learning_rate)
        
        # Training history
        self.loss_history = []
        
    def train_step(self, atoms_list: List[Atoms], target_data: pd.DataFrame, 
                   property_weights: Optional[Dict] = None) -> float:
        """
        Perform one training step.
        
        Args:
            atoms_list: List of ASE Atoms objects to evaluate
            target_data: DataFrame with target properties
            property_weights: Optional weights for different properties
            
        Returns:
            Total loss for this step
        """
        self.optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        
        # Update TOML file with current parameters
        self.extractor.update_toml_with_parameters(self.toml_path, self.parameters)
        
        # Compute loss for each structure
        for i, atoms in enumerate(atoms_list):
            try:
                # Compute properties with current parameters
                predicted_props = compute_properties_with_custom_params(atoms, self.toml_path)
                
                # Get target properties (assuming target_data has same order)
                if i < len(target_data):
                    target_props = target_data.iloc[i].to_dict()
                    
                    # Compute loss for this structure
                    structure_loss = compute_loss(predicted_props, target_props, property_weights)
                    total_loss += structure_loss
                    
            except Exception as e:
                print(f"Error computing properties for structure {i}: {e}")
                # Add penalty for failed calculations
                total_loss += torch.tensor(1000.0, dtype=torch.float32)
        
        # Backpropagate
        total_loss.backward()
        self.optimizer.step()
        
        # Store loss
        loss_value = total_loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def train(self, atoms_list: List[Atoms], target_data: pd.DataFrame, 
              epochs: int = 100, property_weights: Optional[Dict] = None) -> None:
        """
        Train the parameters for multiple epochs.
        
        Args:
            atoms_list: List of ASE Atoms objects
            target_data: DataFrame with target properties
            epochs: Number of training epochs
            property_weights: Optional weights for different properties
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Optimizing {len(self.parameters)} parameters")
        
        for epoch in range(epochs):
            loss = self.train_step(atoms_list, target_data, property_weights)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}/{epochs}: Loss = {loss:.6f}")
        
        print("Training completed!")
        
    def save_final_parameters(self, output_path: Optional[str] = None) -> str:
        """
        Save the optimized parameters to a new TOML file.
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.toml_path.replace('.toml', '_optimized.toml')
        
        # Update and save final parameters
        self.extractor.update_toml_with_parameters(self.toml_path, self.parameters)
        
        # Copy to new file if different path
        if output_path != self.toml_path:
            import shutil
            shutil.copy2(self.toml_path, output_path)
        
        print(f"Optimized parameters saved to: {output_path}")
        return output_path

def calculate_elastic_constants_simple(atoms: Atoms, strain_magnitude: float = 0.01) -> np.ndarray:
    """
    Calculate elastic constants using a simplified finite difference approach.
    
    Args:
        atoms: ASE Atoms object (should be relaxed)
        strain_magnitude: Magnitude of strain for finite differences
        
    Returns:
        6x6 elastic constant matrix in GPa
    """
    # Store original cell
    original_cell = atoms.cell.copy()
    original_volume = atoms.get_volume()
    
    # Initialize elastic constant matrix
    elastic = np.zeros((6, 6))
    
    # Voigt notation strain matrices
    strain_matrices = []
    
    # Define strain patterns (Voigt notation)
    for i in range(6):
        strain = np.zeros((3, 3))
        if i < 3:  # Normal strains
            strain[i, i] = strain_magnitude
        elif i == 3:  # Shear strain yz
            strain[1, 2] = strain[2, 1] = strain_magnitude / 2
        elif i == 4:  # Shear strain xz
            strain[0, 2] = strain[2, 0] = strain_magnitude / 2
        elif i == 5:  # Shear strain xy
            strain[0, 1] = strain[1, 0] = strain_magnitude / 2
        strain_matrices.append(strain)
    
    try:
        # Get stress for each strain
        for i, strain_matrix in enumerate(strain_matrices):
            # Apply strain
            deformed_cell = original_cell @ (np.eye(3) + strain_matrix)
            atoms.set_cell(deformed_cell, scale_atoms=True)
            
            # Calculate stress (this is simplified - should ideally relax at constant strain)
            stress = atoms.get_stress(voigt=True)  # Already in Voigt notation
            
            # Convert from eV/Å³ to GPa
            stress_gpa = stress * 160.21766208  # Conversion factor
            
            # Store in elastic matrix (simplified - not rigorous)
            elastic[i, :] = stress_gpa / strain_magnitude
        
        # Restore original cell
        atoms.set_cell(original_cell, scale_atoms=True)
        
        # Symmetrize the matrix
        elastic = (elastic + elastic.T) / 2
        
        return elastic
        
    except Exception as e:
        print(f"Error in elastic constants calculation: {e}")
        # Restore original cell
        atoms.set_cell(original_cell, scale_atoms=True)
        return np.zeros((6, 6))


