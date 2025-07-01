#!/usr/bin/env python3
"""Test script for improved Bayesian optimization on Si2"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bayes_h import GeneralParameterBayesian, BayesianConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def main():
    print("Testing improved Bayesian optimization for Si2")
    print("=" * 50)
    
    # Configuration
    CONFIG_DIR = Path("config")
    BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"
    
    # Bayesian optimization configuration
    # Using more conservative settings for initial test
    config = BayesianConfig(
        n_calls=50,           # Reduced for faster test
        n_initial_points=10,  # Reduced but proportional
        acq_func="EI",        # Expected Improvement
        xi=0.01               # Small exploration bonus
    )
    
    print(f"Configuration:")
    print(f"  Function evaluations: {config.n_calls}")
    print(f"  Initial random points: {config.n_initial_points}")
    print(f"  Acquisition function: {config.acq_func}")
    print(f"  Exploration parameter: {config.xi}")
    print()
    
    try:
        # Initialize Bayesian optimizer
        bayes = GeneralParameterBayesian("Si2", str(BASE_PARAM_FILE), config=config)
        
        print(f"Optimizing {len(bayes.parameter_bounds)} parameters:")
        for bound in bayes.parameter_bounds:
            print(f"  {bound.name}: [{bound.min_val:.3f}, {bound.max_val:.3f}] (default: {bound.default_val:.3f})")
        print()
        
        # Run optimization
        print("Starting optimization...")
        best_parameters = bayes.optimize()
        
        # Results
        print(f"\nOptimization completed!")
        print(f"Best RMSE: {bayes.best_fitness:.6f}")
        print(f"Failed evaluations: {bayes.failed_evaluations}")
        print(f"Total function evaluations: {bayes.call_count}")
        print()
        
        if best_parameters:
            print("Best parameters:")
            for param_name, value in best_parameters.items():
                print(f"  {param_name}: {value:.6f}")
                
            # Save results
            bayes.save_best_parameters(bayes.system_config.optimized_params_file)
            bayes.save_fitness_history(bayes.system_config.fitness_history_file)
            print(f"\nResults saved to:")
            print(f"  Parameters: {bayes.system_config.optimized_params_file}")
            print(f"  History: {bayes.system_config.fitness_history_file}")
        else:
            print("No valid parameters found!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
