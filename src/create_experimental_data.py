#!/usr/bin/env python3
"""
Script to create experimental reference data files for solid-state systems.
This prevents the fallback data generation when optimizing lattice constants.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import create_experimental_reference_data, list_available_systems

def main():
    """Create experimental reference data for all solid-state systems"""
    print("Creating experimental reference data files...")
    
    # Get all available systems
    systems = list_available_systems()
    
    # Filter for solid-state systems with lattice constants
    solid_state_systems = []
    for system in systems:
        try:
            from config import get_system_config, CalculationType
            config = get_system_config(system)
            if (config.system_type.value == "solid_state" and 
                config.calculation_type == CalculationType.LATTICE_CONSTANTS):
                solid_state_systems.append(system)
        except Exception as e:
            print(f"Warning: Could not check system {system}: {e}")
    
    print(f"Found {len(solid_state_systems)} solid-state systems: {solid_state_systems}")
    
    # Create reference data for each system
    for system in solid_state_systems:
        try:
            print(f"\nCreating reference data for {system}...")
            ref_file = create_experimental_reference_data(system)
            print(f"✓ Successfully created {ref_file}")
        except Exception as e:
            print(f"✗ Failed to create reference data for {system}: {e}")
    
    print("\nDone! You can now run optimization without fallback data generation.")

if __name__ == "__main__":
    main()
