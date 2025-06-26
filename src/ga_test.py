#!/usr/bin/env python3
"""
Interactive test script for the genetic algorithm parameter optimization.
"""

import sys
from pathlib import Path
import time
import pandas as pd
from ga import TBLiteParameterGA, GAConfig
from ga_analysis import GAAnalyzer

def test_quick():
    """Quick test with minimal parameters"""
    print("Running quick test (8 individuals, 5 generations)...")
    
    config = GAConfig(
        population_size=8,
        generations=5,
        mutation_rate=0.15,
        max_workers=2
    )
    
    ga = TBLiteParameterGA(
        base_param_file="gfn1-base.toml",
        config=config,
        train_fraction=0.8  # Use 80% for training, 20% for testing
    )
    
    print(f"Training on {len(ga.reference_data)} points")
    print(f"Will test on {len(ga.test_reference_data)} points")
    
    start_time = time.time()
    best_individual = ga.optimize()
    elapsed = time.time() - start_time
    
    if best_individual:
        print(f"\nOptimization completed in {elapsed:.1f}s")
        print(f"Best training fitness: {best_individual.fitness:.6f}")
        
        # Save results
        ga.save_best_parameters("ga-optimized-quick.toml")
        ga.save_fitness_history("ga-fitness-quick.csv")
        print("Results saved to ga-optimized-quick.toml and ga-fitness-quick.csv")
        
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

def test_full():
    """Full optimization run"""
    print("Running full optimization (30 individuals, 50 generations)...")
    
    config = GAConfig(
        population_size=30,
        generations=50,
        mutation_rate=0.05,
        crossover_rate=0.8,
        max_workers=4
    )
    
    ga = TBLiteParameterGA(
        base_param_file="gfn1-base.toml",
        config=config,
        train_fraction=0.8
    )
    
    print(f"Training on {len(ga.reference_data)} points")
    print(f"Will test on {len(ga.test_reference_data)} points")
    print("This may take 30-60 minutes...")
    
    start_time = time.time()
    best_individual = ga.optimize()
    elapsed = time.time() - start_time
    
    if best_individual:
        print(f"\nOptimization completed in {elapsed/60:.1f} minutes")
        print(f"Best training fitness: {best_individual.fitness:.6f}")
        
        # Save results
        ga.save_best_parameters("ga-optimized-full.toml")
        ga.save_fitness_history("ga-fitness-full.csv")
        print("Results saved to ga-optimized-full.toml and ga-fitness-full.csv")
        
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

def analyze_results():
    """Analyze GA results"""
    print("Analyzing GA optimization results...")
    
    # Check for existing results with various naming patterns
    result_sets = []
    
    # Pattern 1: Current naming convention
    for suffix in ["-quick", "-full"]:
        param_file = f"ga-optimized{suffix}.toml"
        fitness_file = f"ga-fitness{suffix}.csv"
        
        if Path(param_file).exists() and Path(fitness_file).exists():
            result_sets.append((param_file, fitness_file, f"GA{suffix}"))
    
    # Pattern 2: Legacy test naming
    if Path("test_ga_params.toml").exists() and Path("test_ga_fitness.csv").exists():
        result_sets.append(("test_ga_params.toml", "test_ga_fitness.csv", "Legacy-test"))
    
    # Pattern 3: Old full optimization naming
    if Path("ga_optimized_params.toml").exists() and Path("ga_fitness_history.csv").exists():
        result_sets.append(("ga_optimized_params.toml", "ga_fitness_history.csv", "Full-optimization"))
    
    if not result_sets:
        print("No GA results found. Available files:")
        for f in Path(".").glob("*.toml"):
            if "ga" in f.name.lower() or "test" in f.name.lower():
                print(f"  - {f.name}")
        for f in Path(".").glob("*.csv"):
            if "ga" in f.name.lower() or "test" in f.name.lower() or "fitness" in f.name.lower():
                print(f"  - {f.name}")
        return False
    
    for param_file, fitness_file, name in result_sets:
        print(f"\nAnalyzing {name}: {param_file}")
        
        try:
            # Create analyzer with correct argument names
            analyzer = GAAnalyzer(
                ga_fitness_file=fitness_file,
                ga_params_file=param_file,
                base_params_file="gfn1-base.toml"
            )
            
            # Generate analysis plots
            suffix = name.lower().replace("-", "_")
            
            # Ensure figures directory exists
            Path("figures").mkdir(exist_ok=True)
            
            analyzer.plot_fitness_evolution(save_path=f"figures/ga_fitness_{suffix}.png")
            analyzer.compare_h2_curves(save_path=f"figures/ga_h2_comparison_{suffix}.png")
            analyzer.analyze_parameter_changes(save_path=f"figures/ga_param_changes_{suffix}.png")
            
            print(f"Analysis plots saved to figures/ga_*_{suffix}.png")
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    return True

def compare_with_tblite_fit():
    """Compare GA results with tblite fit"""
    print("Comparing GA optimization with tblite fit...")
    
    # Check for any GA optimization results
    ga_result_files = []
    
    # Check different naming patterns
    for param_file in ["ga-optimized-full.toml", "ga-optimized-quick.toml", 
                      "test_ga_params.toml", "ga_optimized_params.toml"]:
        if Path(param_file).exists():
            ga_result_files.append(param_file)
    
    if not ga_result_files:
        print("No GA results found. Available parameter files:")
        for f in Path(".").glob("*.toml"):
            if "ga" in f.name.lower():
                print(f"  - {f.name}")
        return False
    
    # Check for tblite fit results
    tblite_fit_file = "fitpar.toml"
    if not Path(tblite_fit_file).exists():
        print(f"No tblite fit results found ({tblite_fit_file})")
        print("You can run tblite fit first, or this comparison will just show GA vs base parameters")
    
    print(f"Found GA results: {ga_result_files}")
    
    # For now, just report what files were found
    for ga_file in ga_result_files:
        print(f"\nGA result: {ga_file}")
        try:
            # Show file modification time
            import os
            mtime = os.path.getmtime(ga_file)
            import datetime
            print(f"  Modified: {datetime.datetime.fromtimestamp(mtime)}")
            
            # Show file size
            size = os.path.getsize(ga_file) / 1024  # KB
            print(f"  Size: {size:.1f} KB")
            
        except Exception as e:
            print(f"  Error reading file info: {e}")
    
    if Path(tblite_fit_file).exists():
        print(f"\nTBLite fit result: {tblite_fit_file}")
        try:
            import os
            mtime = os.path.getmtime(tblite_fit_file)
            import datetime
            print(f"  Modified: {datetime.datetime.fromtimestamp(mtime)}")
            size = os.path.getsize(tblite_fit_file) / 1024  # KB
            print(f"  Size: {size:.1f} KB")
        except Exception as e:
            print(f"  Error reading file info: {e}")
    
    # TODO: Implement detailed comparison
    print("\nDetailed comparison functionality could be implemented here:")
    print("- Compare parameter values between GA and tblite fit")
    print("- Compare H2 curve RMSE vs CCSD for both methods") 
    print("- Compare optimization time and convergence")
    
    return True

def show_train_test_split_info():
    """Show information about the train/test split"""
    print("Train/Test Split Information:")
    print("=" * 50)
    
    config = GAConfig(population_size=2, generations=1, max_workers=1)
    ga = TBLiteParameterGA(
        base_param_file="gfn1-base.toml",
        config=config,
        train_fraction=0.8
    )
    
    print(f"Total data points: {len(ga.full_reference_data)}")
    print(f"Training points: {len(ga.reference_data)} ({80}%)")
    print(f"Test points: {len(ga.test_reference_data)} ({20}%)")
    print(f"\nTraining distance range: {ga.train_distances.min():.3f} - {ga.train_distances.max():.3f} Å")
    print(f"Test distance range: {ga.test_reference_data['Distance'].min():.3f} - {ga.test_reference_data['Distance'].max():.3f} Å")
    print(f"\nTraining energy range: {ga.train_energies.min():.6f} - {ga.train_energies.max():.6f} Hartree")
    print(f"Test energy range: {ga.test_reference_data['Energy'].min():.6f} - {ga.test_reference_data['Energy'].max():.6f} Hartree")
    
    # Show some training points
    print(f"\nFirst 5 training points:")
    print(ga.reference_data.head())
    
    print(f"\nFirst 5 test points:")
    print(ga.test_reference_data.head())

def visualize_train_test_split():
    """Create plots showing the train/test split"""
    print("Creating train/test split visualization...")
    
    try:
        from ga_analysis import GAAnalyzer
        
        # Create a dummy analyzer just for the train/test split plot
        # We don't need actual GA results for this visualization
        try:
            # Try to find any existing fitness file for the constructor
            fitness_file = None
            param_file = None
            
            for suffix in ["-quick", "-full", ""]:
                test_fitness = f"ga-fitness{suffix}.csv"
                test_params = f"ga-optimized{suffix}.toml"
                if Path(test_fitness).exists() and Path(test_params).exists():
                    fitness_file = test_fitness
                    param_file = test_params
                    break
            
            if fitness_file and param_file:
                analyzer = GAAnalyzer(fitness_file, param_file)
                analyzer.plot_train_test_split(save_path="figures/train_test_split.png")
            else:
                # Create minimal files for the plot
                print("No GA results found, creating standalone train/test split visualization...")
                
                # Import the method directly
                from ga_analysis import GAAnalyzer
                import tempfile
                import pandas as pd
                
                # Create dummy files
                dummy_fitness = pd.DataFrame({'generation': [0], 'best_fitness': [0.5], 'avg_fitness': [0.4], 'std_fitness': [0.1]})
                dummy_params = {"dummy": "value"}
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    dummy_fitness.to_csv(f.name, index=False)
                    fitness_temp = f.name
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                    import toml
                    toml.dump(dummy_params, f)
                    params_temp = f.name
                
                analyzer = GAAnalyzer(fitness_temp, params_temp)
                analyzer.plot_train_test_split(save_path="figures/train_test_split.png")
                
                # Clean up
                import os
                os.unlink(fitness_temp)
                os.unlink(params_temp)
        
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError:
        print("Error: Could not import ga_analysis module")

def main():
    """Interactive menu for GA testing"""
    
    while True:
        print("\n" + "="*60)
        print("TBLite Genetic Algorithm Parameter Optimization")
        print("="*60)
        print("1. Quick test (8 individuals, 5 generations)")
        print("2. Full optimization (30 individuals, 50 generations)")
        print("3. Analyze results")
        print("4. Compare with tblite fit")
        print("5. Show train/test split info")
        print("6. Visualize train/test split")
        print("7. Exit")
        print()
        
        try:
            choice = input("Choose an option (1-7): ").strip()
            
            if choice == '1':
                test_quick()
            elif choice == '2':
                test_full()
            elif choice == '3':
                analyze_results()
            elif choice == '4':
                compare_with_tblite_fit()
            elif choice == '5':
                show_train_test_split_info()
            elif choice == '6':
                visualize_train_test_split()
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 
