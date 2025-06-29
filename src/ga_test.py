#!/usr/bin/env python3
"""
General test script for parameter optimization on any molecular system.
Uses the system configuration approach to be plug-and-play.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from ga import GeneralParameterGA, GAConfig
from calc import GeneralCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod
from data_extraction import GFN1ParameterExtractor
from config import get_system_config, list_available_systems, print_all_systems, print_system_info

# Portable paths
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"


def test_quick(system_name: str):
    """Quick test with minimal parameters"""
    print(f"Running quick {system_name} test (8 individuals, 5 generations)...")
    
    config = GAConfig(
        population_size=8,
        generations=5,
        mutation_rate=0.15,
        max_workers=2
    )
    
    ga = GeneralParameterGA(
        system_name=system_name,
        base_param_file=str(BASE_PARAM_FILE),
        config=config,
        train_fraction=0.8
    )
    
    print(f"Training on {len(ga.reference_data)} points")
    print(f"Will test on {len(ga.test_reference_data)} points")
    
    start_time = time.time()
    best_individual = ga.optimize()
    elapsed = time.time() - start_time
    
    if best_individual:
        print(f"\nOptimization completed in {elapsed:.1f}s")
        print(f"Best training fitness: {best_individual.fitness:.6f}")
        
        # Save results with system name
        param_file = f"ga-{system_name.lower()}-optimized-quick.toml"
        fitness_file = f"ga-{system_name.lower()}-fitness-quick.csv"
        
        ga.save_best_parameters(param_file)
        ga.save_fitness_history(fitness_file)
        print(f"Results saved to {param_file} and {fitness_file}")
        
        # Show test performance
        test_results = ga.evaluate_test_performance(best_individual)
        print(f"\nGeneralization to test set:")
        print(f"Test RMSE: {test_results['test_rmse']:.6f}")
        print(f"Test fitness: {test_results['test_fitness']:.6f}")
        print(f"Train vs Test fitness ratio: {best_individual.fitness / test_results['test_fitness']:.3f}")
        
        return True
    else:
        print("Optimization failed!")
        return False


def test_full(system_name: str):
    """Full optimization run"""
    print(f"Running full {system_name} optimization (20 individuals, 25 generations)...")
    
    config = GAConfig(
        population_size=20,
        generations=25,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_workers=4
    )
    
    ga = GeneralParameterGA(
        system_name=system_name,
        base_param_file=str(BASE_PARAM_FILE),
        config=config,
        train_fraction=0.8
    )
    
    print(f"Training on {len(ga.reference_data)} points")
    print(f"Will test on {len(ga.test_reference_data)} points")
    print("This may take 20-60 minutes...")
    
    start_time = time.time()
    best_individual = ga.optimize()
    elapsed = time.time() - start_time
    
    if best_individual:
        print(f"\nOptimization completed in {elapsed/60:.1f} minutes")
        print(f"Best training fitness: {best_individual.fitness:.6f}")
        
        # Save results with system name
        param_file = f"ga-{system_name.lower()}-optimized-full.toml"
        fitness_file = f"ga-{system_name.lower()}-fitness-full.csv"
        
        ga.save_best_parameters(param_file)
        ga.save_fitness_history(fitness_file)
        print(f"Results saved to {param_file} and {fitness_file}")
        
        # Show detailed test performance
        test_results = ga.evaluate_test_performance(best_individual)
        print(f"\nGeneralization to test set:")
        print(f"Test RMSE: {test_results['test_rmse']:.6f}")
        print(f"Test MAE: {test_results['test_mae']:.6f}")
        print(f"Test Max Error: {test_results['test_max_error']:.6f}")
        print(f"Test fitness: {test_results['test_fitness']:.6f}")
        print(f"Train vs Test fitness ratio: {best_individual.fitness / test_results['test_fitness']:.3f}")
        
        if best_individual.fitness / test_results['test_fitness'] > 1.2:
            print("WARNING: Significant overfitting detected (train >> test performance)")
        elif best_individual.fitness / test_results['test_fitness'] < 0.8:
            print("NOTE: Model generalizes better than expected (test > train performance)")
        else:
            print("Good generalization: similar train and test performance")
        
        return True
    else:
        print("Optimization failed!")
        return False


def generate_reference_data(system_name: str):
    """Generate reference dissociation curve for a system"""
    print(f"Generating {system_name} reference dissociation curve...")
    
    system_config = get_system_config(system_name)
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "curves").mkdir(parents=True, exist_ok=True)
    
    # Generate with GFN1-xTB as reference
    calc_config = CalcConfig(method=CalcMethod.GFN1_XTB)
    calculator = GeneralCalculator(calc_config, system_config)
    generator = DissociationCurveGenerator(calculator)
    
    print(f"Calculating {system_name} curve...")
    try:
        ref_data = generator.generate_curve(
            save=True, filename=system_config.reference_data_file
        )
        
        print(f"Reference data generated and saved to {system_config.reference_data_file}")
        print(f"Data shape: {ref_data.shape}")
        
        if system_config.bond_range:
            print(f"Distance range: {system_config.bond_range[0]:.2f} - {system_config.bond_range[1]:.2f} Å")
        
        print(f"Energy range: {ref_data['Energy'].min():.6f} - {ref_data['Energy'].max():.6f} Hartree")
        
        # Find minimum
        min_idx = np.argmin(ref_data['Energy'])
        min_distance = ref_data['Distance'].iloc[min_idx]
        min_energy = ref_data['Energy'].iloc[min_idx]
        print(f"Minimum energy: {min_energy:.6f} Hartree at {min_distance:.2f} Å")
        
        return True
    except Exception as e:
        print(f"Error generating reference data: {e}")
        return False


def show_parameter_info(system_name: str):
    """Show information about extracted parameters for a system"""
    print(f"{system_name} Parameter Extraction Information:")
    print("=" * 50)
    
    try:
        system_config = get_system_config(system_name)
        
        # Extract parameters
        extractor = GFN1ParameterExtractor(BASE_PARAM_FILE)
        system_params = extractor.extract_defaults_dict(system_config.elements)
        print(f"Extracted {len(system_params)} {system_name}-relevant parameters:")
        
        # Group by type
        param_groups = {}
        for name, value in system_params.items():
            if 'hamiltonian' in name:
                group = 'Hamiltonian'
            elif any(f'element.{elem}' in name for elem in system_config.elements):
                group = f'Element ({", ".join(system_config.elements)})'
            else:
                group = 'Other'
            
            if group not in param_groups:
                param_groups[group] = []
            param_groups[group].append((name, value))
        
        for group, params in param_groups.items():
            print(f"\n{group} Parameters ({len(params)}):")
            for name, value in params[:5]:  # Show first 5
                print(f"  {name}: {value:.6f}")
            if len(params) > 5:
                print(f"  ... and {len(params) - 5} more")
        
        print(f"\nParameter bounds are automatically generated based on:")
        print("- Parameter type (energy levels, Slater exponents, etc.)")
        print("- Physical constraints (positivity, reasonable ranges)")
        print("- Default values from base parameter file")
        print(f"- {system_name}-specific constraints (e.g., bond parameters)")
        
    except Exception as e:
        print(f"Error extracting parameters: {e}")


def show_train_test_split_info(system_name: str):
    """Show information about the train/test split for a system"""
    print(f"{system_name} Train/Test Split Information:")
    print("=" * 50)
    
    config = GAConfig(population_size=2, generations=1, max_workers=1)
    ga = GeneralParameterGA(
        system_name=system_name,
        base_param_file=str(BASE_PARAM_FILE),
        config=config,
        train_fraction=0.8
    )
    
    print(f"Total data points: {len(ga.full_reference_data)}")
    print(f"Training points: {len(ga.reference_data)} ({80}%)")
    print(f"Test points: {len(ga.test_reference_data)} ({20}%)")
    print(f"Optimization parameters: {len(ga.parameter_bounds)}")
    print(f"\nTraining distance range: {ga.train_distances.min():.3f} - {ga.train_distances.max():.3f} Å")
    print(f"Test distance range: {ga.test_reference_data['Distance'].min():.3f} - {ga.test_reference_data['Distance'].max():.3f} Å")
    print(f"\nTraining energy range: {ga.train_energies.min():.6f} - {ga.train_energies.max():.6f} Hartree")
    print(f"Test energy range: {ga.test_reference_data['Energy'].min():.6f} - {ga.test_reference_data['Energy'].max():.6f} Hartree")
    
    # Show some training points
    print(f"\nFirst 5 training points:")
    print(ga.reference_data.head())
    
    print(f"\nFirst 5 test points:")
    print(ga.test_reference_data.head())


def check_reference_data(system_name: str):
    """Check if reference data exists and show info"""
    print(f"Checking {system_name} Reference Data:")
    print("=" * 30)
    
    system_config = get_system_config(system_name)
    ref_file = Path(system_config.reference_data_file)
    
    if ref_file.exists():
        try:
            data = pd.read_csv(ref_file)
            print(f"✓ Reference data found: {ref_file}")
            print(f"  Data points: {len(data)}")
            print(f"  Distance range: {data['Distance'].min():.2f} - {data['Distance'].max():.2f} Å")
            print(f"  Energy range: {data['Energy'].min():.6f} - {data['Energy'].max():.6f} Hartree")
            
            # Find minimum
            min_idx = data['Energy'].idxmin()
            min_distance = data['Distance'].iloc[min_idx]
            min_energy = data['Energy'].iloc[min_idx]
            print(f"  Minimum: {min_energy:.6f} Hartree at {min_distance:.2f} Å")
            
            return True
        except Exception as e:
            print(f"✗ Error reading reference data: {e}")
            return False
    else:
        print(f"✗ Reference data not found: {ref_file}")
        print("  Run 'Generate reference data' option to create reference data first")
        return False


def select_system():
    """Interactive system selection"""
    systems = list_available_systems()
    
    print("\nAvailable Systems:")
    print("-" * 20)
    for i, system in enumerate(systems, 1):
        system_config = get_system_config(system)
        if system_config.bond_range:
            range_str = f"({system_config.bond_range[0]:.1f}-{system_config.bond_range[1]:.1f} Å)"
        else:
            range_str = ""
        elements_str = ", ".join(system_config.elements)
        print(f"{i:2}. {system:8} - {elements_str:6} {range_str}")
    
    while True:
        try:
            choice = input(f"\nSelect system (1-{len(systems)}) or name: ").strip()
            
            # Try as number first
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(systems):
                    return systems[idx]
                else:
                    print(f"Invalid choice. Please select 1-{len(systems)}")
                    continue
            
            # Try as system name
            if choice in systems:
                return choice
            
            print(f"Invalid system '{choice}'. Please try again.")
            
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def main():
    """Interactive menu for general parameter optimization testing"""
    
    current_system = None
    
    while True:
        print("\n" + "="*60)
        print("General Parameter Optimization Testing")
        print("="*60)
        
        if current_system:
            print(f"Current system: {current_system}")
            print_system_info(current_system)
            print()
        
        print("System Selection:")
        print("1. Select/change system")
        print("2. Show all available systems")
        print()
        
        if current_system:
            print("System Operations:")
            print("3. Show parameter extraction info")
            print("4. Generate reference data")
            print("5. Check reference data")
            print("6. Show train/test split info")
            print()
            print("Optimization:")
            print("7. Quick test (8 individuals, 5 generations)")
            print("8. Full optimization (20 individuals, 25 generations)")
            print()
        
        print("9. Exit")
        print()
        
        try:
            choice = input("Choose an option: ").strip()
            
            if choice == '1':
                selected = select_system()
                if selected:
                    current_system = selected
                    print(f"Selected system: {current_system}")
                    
            elif choice == '2':
                print_all_systems()
                
            elif choice == '3' and current_system:
                show_parameter_info(current_system)
                
            elif choice == '4' and current_system:
                generate_reference_data(current_system)
                
            elif choice == '5' and current_system:
                check_reference_data(current_system)
                
            elif choice == '6' and current_system:
                if check_reference_data(current_system):
                    show_train_test_split_info(current_system)
                else:
                    print("Please generate reference data first")
                    
            elif choice == '7' and current_system:
                if check_reference_data(current_system):
                    test_quick(current_system)
                else:
                    print("Please generate reference data first")
                    
            elif choice == '8' and current_system:
                if check_reference_data(current_system):
                    test_full(current_system)
                else:
                    print("Please generate reference data first")
                    
            elif choice == '9':
                print("Goodbye!")
                break
                
            elif not current_system and choice in ['3', '4', '5', '6', '7', '8']:
                print("Please select a system first (option 1)")
                
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Allow direct system specification from command line
    if len(sys.argv) > 1:
        system_name = sys.argv[1]
        print(f"Running quick test for {system_name}")
        test_quick(system_name)
    else:
        main()
