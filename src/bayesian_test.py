"""
Test script for Bayesian Optimization of TBLite parameters
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from pathlib import Path

from bayesian import TBLiteParameterBayesian, BayesianConfig

# Portable paths - automatically finds project root from current working directory
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
TESTS_DIR = PROJECT_ROOT / "tests"

# Configuration files
BASE_PARAM_FILE = CONFIG_DIR / "gfn1-base.toml"

# Test output files
TEST_BAYESIAN_PARAMS = TESTS_DIR / "test_bayesian_params.toml"
TEST_BAYESIAN_FITNESS = TESTS_DIR / "test_bayesian_fitness.csv"
TEST_CONVERGENCE_PLOT = TESTS_DIR / "test_bayesian_convergence.png"

# Full optimization output files
BAYESIAN_OPTIMIZED_PARAMS = RESULTS_DIR / "parameters" / "bayesian_optimized_params.toml"
BAYESIAN_FITNESS_HISTORY = RESULTS_DIR / "fitness" / "bayesian_fitness_history.csv"
BAYESIAN_CONVERGENCE_PLOT = RESULTS_DIR / "plots" / "bayesian_convergence.png"

# Comparison output files
ACQUISITION_COMPARISON_PLOT = RESULTS_DIR / "plots" / "bayesian_acquisition_comparison.png"
ACQUISITION_COMPARISON_CSV = RESULTS_DIR / "fitness" / "bayesian_acquisition_comparison.csv"
DETAILED_CONVERGENCE_PLOT = RESULTS_DIR / "plots" / "bayesian_detailed_convergence.png"

def quick_test():
    """Quick test with minimal iterations"""
    print("=== Quick Bayesian Optimization Test ===")
    
    config = BayesianConfig(
        n_initial_points=3,
        n_iterations=10,
        acquisition_function="ei",
        random_state=42
    )
    
    print(f"Configuration:")
    print(f"  Initial points: {config.n_initial_points}")
    print(f"  Total iterations: {config.n_iterations}")
    print(f"  Acquisition function: {config.acquisition_function}")
    print(f"  GP kernel: {config.gp_kernel}")
    
    optimizer = TBLiteParameterBayesian(
        base_param_file=str(BASE_PARAM_FILE),
        config=config
    )
    
    print(f"\nOptimizing {len(optimizer.parameter_bounds)} parameters...")
    for bound in optimizer.parameter_bounds:
        print(f"  {bound.name}: [{bound.min_val:.3f}, {bound.max_val:.3f}]")
    
    start_time = time.time()
    best_params = optimizer.optimize()
    elapsed = time.time() - start_time
    
    print(f"\n=== Results ===")
    print(f"Best fitness (RMSE): {optimizer.best_fitness:.6f} Hartree")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Failed evaluations: {optimizer.failed_evaluations}")
    
    # Save results
    optimizer.save_best_parameters(str(TEST_BAYESIAN_PARAMS))
    optimizer.save_fitness_history(str(TEST_BAYESIAN_FITNESS))
    
    # Plot convergence
    plot_convergence(optimizer.fitness_history, str(TEST_CONVERGENCE_PLOT))
    
    return optimizer

def full_optimization():
    """Full optimization run"""
    print("=== Full Bayesian Optimization ===")
    
    config = BayesianConfig(
        n_initial_points=10,
        n_iterations=50,
        acquisition_function="ei",
        random_state=42
    )
    
    optimizer = TBLiteParameterBayesian(
        base_param_file=str(BASE_PARAM_FILE),
        config=config
    )
    
    start_time = time.time()
    best_params = optimizer.optimize()
    elapsed = time.time() - start_time
    
    print(f"\n=== Results ===")
    print(f"Best fitness (RMSE): {optimizer.best_fitness:.6f} Hartree")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Iterations per second: {len(optimizer.fitness_history)/elapsed:.2f}")
    
    # Save results
    optimizer.save_best_parameters(str(BAYESIAN_OPTIMIZED_PARAMS))
    optimizer.save_fitness_history(str(BAYESIAN_FITNESS_HISTORY))
    
    # Plot convergence
    plot_convergence(optimizer.fitness_history, str(BAYESIAN_CONVERGENCE_PLOT))
    
    return optimizer

def compare_acquisition_functions():
    """Compare different acquisition functions"""
    print("=== Comparing Acquisition Functions ===")
    
    acquisition_functions = ["ei", "pi", "ucb"]
    results = {}
    
    for acq_func in acquisition_functions:
        print(f"\nTesting {acq_func.upper()}...")
        
        config = BayesianConfig(
            n_initial_points=5,
            n_iterations=20,
            acquisition_function=acq_func,
            random_state=42
        )
        
        optimizer = TBLiteParameterBayesian(
            base_param_file=str(BASE_PARAM_FILE),
            config=config
        )
        
        start_time = time.time()
        best_params = optimizer.optimize()
        elapsed = time.time() - start_time
        
        results[acq_func] = {
            'best_fitness': optimizer.best_fitness,
            'elapsed': elapsed,
            'fitness_history': optimizer.fitness_history.copy(),
            'failed_evaluations': optimizer.failed_evaluations
        }
        
        print(f"  Best fitness: {optimizer.best_fitness:.6f}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Failed evaluations: {optimizer.failed_evaluations}")
    
    # Plot comparison
    plot_acquisition_comparison(results, str(ACQUISITION_COMPARISON_PLOT))
    
    # Save comparison results
    comparison_df = pd.DataFrame({
        acq_func: {
            'best_fitness': results[acq_func]['best_fitness'],
            'time_seconds': results[acq_func]['elapsed'],
            'failed_evaluations': results[acq_func]['failed_evaluations']
        }
        for acq_func in acquisition_functions
    }).T
    
    comparison_df.to_csv(str(ACQUISITION_COMPARISON_CSV))
    print(f"\nComparison saved to {ACQUISITION_COMPARISON_CSV}")
    print(comparison_df)
    
    return results

def convergence_analysis():
    """Analyze convergence behavior"""
    print("=== Convergence Analysis ===")
    
    config = BayesianConfig(
        n_initial_points=10,
        n_iterations=100,
        acquisition_function="ei",
        patience=20,
        random_state=42
    )
    
    optimizer = TBLiteParameterBayesian(
        base_param_file=str(BASE_PARAM_FILE),
        config=config
    )
    
    start_time = time.time()
    best_params = optimizer.optimize()
    elapsed = time.time() - start_time
    
    # Analyze convergence
    fitness_history = optimizer.fitness_history
    
    # Find improvement points
    improvements = []
    best_so_far = float('inf')
    for i, fitness in enumerate(fitness_history):
        if fitness < best_so_far:
            best_so_far = fitness
            improvements.append((i, fitness))
    
    print(f"\nConvergence Analysis:")
    print(f"  Total iterations: {len(fitness_history)}")
    print(f"  Number of improvements: {len(improvements)}")
    print(f"  Final best fitness: {optimizer.best_fitness:.6f}")
    print(f"  First improvement at iteration: {improvements[0][0] if improvements else 'None'}")
    print(f"  Last improvement at iteration: {improvements[-1][0] if improvements else 'None'}")
    
    if len(improvements) > 1:
        improvement_intervals = [improvements[i+1][0] - improvements[i][0] 
                               for i in range(len(improvements)-1)]
        print(f"  Average improvement interval: {np.mean(improvement_intervals):.1f} iterations")
        print(f"  Std improvement interval: {np.std(improvement_intervals):.1f} iterations")
    
    # Plot detailed convergence
    plot_detailed_convergence(fitness_history, improvements, str(DETAILED_CONVERGENCE_PLOT))
    
    return optimizer

def plot_convergence(fitness_history, filename):
    """Plot convergence curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_history)), fitness_history, 'b-', linewidth=2, alpha=0.7)
    
    # Plot best so far
    best_so_far = []
    current_best = float('inf')
    for fitness in fitness_history:
        if fitness < current_best:
            current_best = fitness
        best_so_far.append(current_best)
    
    plt.plot(range(len(best_so_far)), best_so_far, 'r-', linewidth=2, label='Best so far')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (RMSE)')
    plt.title('Bayesian Optimization Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved as {filename}")

def plot_acquisition_comparison(results, filename):
    """Plot comparison of acquisition functions"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    
    for i, (acq_func, result) in enumerate(results.items()):
        fitness_history = result['fitness_history']
        
        # Plot best so far
        best_so_far = []
        current_best = float('inf')
        for fitness in fitness_history:
            if fitness < current_best:
                current_best = fitness
            best_so_far.append(current_best)
        
        plt.plot(range(len(best_so_far)), best_so_far, 
                color=colors[i], linewidth=2, label=f'{acq_func.upper()}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (RMSE)')
    plt.title('Acquisition Function Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Acquisition comparison plot saved as {filename}")

def plot_detailed_convergence(fitness_history, improvements, filename):
    """Plot detailed convergence with improvement markers"""
    plt.figure(figsize=(12, 8))
    
    # Plot all fitness values
    plt.subplot(2, 1, 1)
    plt.plot(range(len(fitness_history)), fitness_history, 'b-', alpha=0.6, label='All evaluations')
    
    # Mark improvements
    if improvements:
        imp_iterations, imp_fitness = zip(*improvements)
        plt.scatter(imp_iterations, imp_fitness, color='red', s=50, zorder=5, label='Improvements')
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (RMSE)')
    plt.title('Detailed Convergence Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot improvement intervals
    plt.subplot(2, 1, 2)
    if len(improvements) > 1:
        imp_iterations, _ = zip(*improvements)
        intervals = [imp_iterations[i+1] - imp_iterations[i] for i in range(len(imp_iterations)-1)]
        plt.bar(range(len(intervals)), intervals, alpha=0.7)
        plt.xlabel('Improvement Number')
        plt.ylabel('Iterations Between Improvements')
        plt.title('Improvement Intervals')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Not enough improvements to analyze', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed convergence plot saved as {filename}")

def main():
    """Main test function with options"""
    print("Bayesian Optimization Test Suite")
    print("=" * 40)
    
    print("\nSelect test to run:")
    print("1. Quick test (3 initial + 10 iterations)")
    print("2. Full optimization (10 initial + 50 iterations)")
    print("3. Compare acquisition functions (EI, PI, UCB)")
    print("4. Convergence analysis (10 initial + 100 iterations)")
    print("5. Run all tests")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            quick_test()
        elif choice == "2":
            full_optimization()
        elif choice == "3":
            compare_acquisition_functions()
        elif choice == "4":
            convergence_analysis()
        elif choice == "5":
            print("Running all tests...")
            quick_test()
            print("\n" + "="*50 + "\n")
            compare_acquisition_functions()
            print("\n" + "="*50 + "\n")
            convergence_analysis()
        else:
            print("Invalid choice. Running quick test...")
            quick_test()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 
