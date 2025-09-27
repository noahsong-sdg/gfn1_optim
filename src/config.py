"""
system config for dissociation curves and lattice stuff
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import numpy as np
import pandas as pd

# Portable paths
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"

class SystemType(Enum):
    """Types of systems that can be optimized"""
    DIATOMIC_MOLECULE = "diatomic"
    SOLID_STATE = "solid"
    SUPERCELL = "supercell"

class CalculationType(Enum):
    """Types of calculations to perform"""
    DISSOCIATION_CURVE = "dissociation"
    BULK = "bulk"
    LATTICE_CONSTANTS = "lattice_constants"

@dataclass
class SystemConfig:
    """Configuration for a specific system"""
    name: str
    system_type: SystemType
    calculation_type: CalculationType
    elements: List[str]
    
    # Core parameters
    spin_multiplicity: int = 0
    
    # For diatomic molecules
    bond_range: Optional[Tuple[float, float]] = None
    num_points: Optional[int] = 100
    
    # For solid state
    crystal_system: Optional[str] = None
    lattice_params: Optional[Dict[str, float]] = None
    
    # For bulk materials
    num_structures: Optional[int] = None
    
    # File paths (auto-generated if not specified)
    reference_data_file: Optional[str] = None
    optimized_params_file: Optional[str] = None
    fitness_history_file: Optional[str] = None
    
    def __post_init__(self):
        """Auto-generate file paths if not specified"""
        if self.reference_data_file is None:
            if self.calculation_type == CalculationType.LATTICE_CONSTANTS:
                pass
            elif self.calculation_type == CalculationType.BULK:
                # Bulk/supercell reference data provided externally
                self.reference_data_file = str(PROJECT_ROOT / "train_structs" / "results.csv")
            else:
                self.reference_data_file = str(PROJECT_ROOT / "test_structs" / "results.csv")
        
        if self.optimized_params_file is None:
            self.optimized_params_file = str(RESULTS_DIR / "parameters" / f"{self.name.lower()}_optimized.toml")
        
        if self.fitness_history_file is None:
            self.fitness_history_file = str(RESULTS_DIR / "fitness" / f"{self.name.lower()}_fitness_history.csv")

# Pre-defined system configurations
SYSTEM_CONFIGS = {
    "H2": SystemConfig(
        name="H2",
        system_type=SystemType.DIATOMIC_MOLECULE,
        calculation_type=CalculationType.DISSOCIATION_CURVE,
        elements=["H"],
        bond_range=(0.4, 4.0),
        num_points=200,
        spin_multiplicity=0  
    ),
    
    "Si2": SystemConfig(
        name="Si2", 
        system_type=SystemType.DIATOMIC_MOLECULE,
        calculation_type=CalculationType.DISSOCIATION_CURVE,
        elements=["Si"],
        bond_range=(1.5, 5.0),
        num_points=500,
        spin_multiplicity=2),  # Triplet?)
    
    "CdS": SystemConfig(
        name="CdS",
        system_type=SystemType.SOLID_STATE,
        calculation_type=CalculationType.LATTICE_CONSTANTS,
        elements=["Cd", "S"],
        crystal_system="wurtzite",
        lattice_params={"a": 4.17, "b": 4.17, "c": 6.78, "alpha": 90, "beta": 90, "gamma": 120, "energy": -12.299729},  # Experimental values from literature
        spin_multiplicity=0
    ),
    "GaN": SystemConfig(
        name="GaN",
        system_type=SystemType.SOLID_STATE,
        calculation_type=CalculationType.LATTICE_CONSTANTS,
        elements=["Ga", "N"],
        crystal_system="wurtzite",
        lattice_params={"a": 3.19, "b": 3.19, "c": 5.19, "alpha": 90, "beta": 90, "gamma": 120, "energy": -12.299729},
        spin_multiplicity=0
    ),

    "big": SystemConfig(
        name="big",
        system_type=SystemType.SUPERCELL,
        calculation_type=CalculationType.BULK,
        elements=["Cd", "S"],
        num_points=200,
        spin_multiplicity=0
    )
}

def get_system_config(system_name: str) -> SystemConfig:
    """Get configuration for a specific system"""
    if system_name not in SYSTEM_CONFIGS:
        available = ", ".join(SYSTEM_CONFIGS.keys())
        raise ValueError(f"System '{system_name}' not found. Available: {available}")
    return SYSTEM_CONFIGS[system_name]

def list_available_systems() -> List[str]:
    """List all available system configurations"""
    return list(SYSTEM_CONFIGS.keys())

def get_systems_by_type(system_type: SystemType) -> List[str]:
    """Get systems of a specific type"""
    return [name for name, config in SYSTEM_CONFIGS.items() 
            if config.system_type == system_type]

def get_calculation_distances(config: SystemConfig) -> np.ndarray:
    """Get distances for calculation based on system config"""
    if config.calculation_type == CalculationType.DISSOCIATION_CURVE:
        if config.bond_range is None:
            raise ValueError(f"Bond range not defined for {config.name}")
        return np.linspace(config.bond_range[0], config.bond_range[1], config.num_points)
    else:
        raise ValueError(f"Distance calculation not implemented for {config.calculation_type}")

def create_molecule_geometry(config: SystemConfig, distance: float) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Create molecule geometry for a given distance"""
    if config.system_type != SystemType.DIATOMIC_MOLECULE:
        raise ValueError(f"Molecule geometry only for diatomic molecules, not {config.system_type}")
    
    if len(config.elements) != 1:
        raise ValueError(f"Diatomic molecule should have 1 element type, got {len(config.elements)}")
    
    element = config.elements[0]
    return [
        (element, (0.0, 0.0, 0.0)),
        (element, (0.0, 0.0, distance))
    ]

def get_isolated_atom_symbol(config: SystemConfig) -> str:
    """Return the atomic symbol for isolated-atom reference of a diatomic system.
    For diatomics we assume both atoms share the same element type (config.elements[0]).
    """
    if not config.elements:
        raise ValueError("SystemConfig.elements is empty")
    return config.elements[0]


def print_system_info(system_name: str):
    """Print detailed information about a system"""
    config = get_system_config(system_name)
    
    print(f"System: {config.name}")
    print(f"Type: {config.system_type.value}")
    print(f"Calculation: {config.calculation_type.value}")
    print(f"Elements: {', '.join(config.elements)}")
    
    if config.system_type == SystemType.DIATOMIC_MOLECULE:
        print(f"Bond range: {config.bond_range[0]:.2f} - {config.bond_range[1]:.2f} Å")
        print(f"Spin multiplicity: {config.spin_multiplicity}")
    elif config.system_type == SystemType.SOLID_STATE:
        print(f"Crystal system: {config.crystal_system}")
        if config.lattice_params:
            params_str = ", ".join([f"{k}={v:.3f}" for k, v in config.lattice_params.items()])
            print(f"Lattice parameters: {params_str} Å")
    
    print(f"Data points: {config.num_points}")
    print(f"Reference data: {config.reference_data_file}")
    print(f"Output parameters: {config.optimized_params_file}")
