"""
PSO Test Script - Integration with h2_v2.py framework
Shows how to use PSO optimizer alongside existing methods
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

from h2_v2 import H2StudyManager, CalcMethod, CalcConfig
from pso import TBLiteParameterPSO, PSOConfig

def quick_pso_test():
    """Quick PSO test with small configuration"""
    print("=== Quick PSO Test ===")
    
    # Small configuration for testing
    pso_config = PSOConfig(
        n_particles=10,
        max_iterations=20,
        max_workers=2,
        patience=5
    )
    
    # Initialize optimizer
    try:
        optimizer = TBLiteParameterPSO(
            base_param_file="gfn1-base.toml",
            config=pso_config
        )
        
        print(f"Initialized PSO with {len(optimizer.parameter_bounds)} parameters")
        print(f"Training data points: {len(optimizer.train_distances)}")
        
        # Run optimization
        start_time = time.time()
        best_params = optimizer.optimize()
        elapsed = time.time() - start_time
        
        print(f"\nPSO completed in {elapsed:.1f}s")
        print(f"Best fitness: {optimizer.global_best_fitness:.6f}")
        
        # Save results
        optimizer.save_best_parameters("pso_quick_params.toml")
        optimizer.save_fitness_history("pso_quick_history.csv")
        
        # Test performance
        test_metrics = optimizer.evaluate_test_performance(best_params)
        print("\nTest Performance:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.6f}")
            
        return True
        
    except Exception as e:
        print(f"PSO test failed: {e}")
        return False

def full_pso_optimization():
    """Full PSO optimization run"""
    print("=== Full PSO Optimization ===")
    
    pso_config = PSOConfig(
        n_particles=30,
        max_iterations=100,
        max_workers=4
    )
    
    optimizer = TBLiteParameterPSO(
        base_param_file="gfn1-base.toml",
        config=pso_config
    )
    
    print("Running full PSO optimization (this may take 30-60 minutes)...")
    best_params = optimizer.optimize()
    
    # Save results
    optimizer.save_best_parameters("pso_optimized_params.toml")
    optimizer.save_fitness_history("pso_fitness_history.csv")
    
    # Evaluate test performance
    test_metrics = optimizer.evaluate_test_performance(best_params)
    print("Final Test Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return best_params

def compare_optimization_methods():
    """Compare PSO with GA and default methods"""
    print("=== Comparing Optimization Methods ===")
    
    # Set up distances for comparison
    distances = np.linspace(0.4, 4.0, 200)  # Smaller for faster comparison
    study = H2StudyManager(distances)
    
    # Add CCSD reference
    ccsd_config = CalcConfig(method=CalcMethod.CCSD, basis="cc-pVTZ")
    study.add_method("CCSD/cc-pVTZ", ccsd_config, "h2_ccsd_comparison.csv")
    
    # Add default GFN1-xTB
    xtb_config = CalcConfig(method=CalcMethod.GFN1_XTB, spin=1)
    study.add_method("GFN1-xTB (default)", xtb_config, "h2_xtb_comparison.csv")
    
    # Add PSO optimized parameters if available
    if Path("pso_optimized_params.toml").exists():
        pso_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file="pso_optimized_params.toml",
            spin=1
        )
        study.add_method("PSO Optimized", pso_config, "h2_pso_comparison.csv")
    
    # Add GA optimized parameters if available
    if Path("ga_optimized_params.toml").exists():
        ga_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file="ga_optimized_params.toml",
            spin=1
        )
        study.add_method("GA Optimized", ga_config, "h2_ga_comparison.csv")
    
    # Generate comparison plot and metrics
    study.plot_comparison('optimization_methods_comparison.png')
    
    print("Comparison complete. Check optimization_methods_comparison.png")

def analyze_pso_convergence():
    """Analyze PSO convergence behavior"""
    if not Path("pso_fitness_history.csv").exists():
        print("No PSO fitness history found. Run optimization first.")
        return
    
    import matplotlib.pyplot as plt
    
    # Load fitness history
    history = pd.read_csv("pso_fitness_history.csv")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(history['iteration'], history['best_fitness'], 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (RMSE)')
    plt.title('PSO Convergence for H₂ Parameter Optimization')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often shows convergence better
    plt.tight_layout()
    plt.savefig('pso_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence analysis saved to pso_convergence.png")
    print(f"Final fitness: {history['best_fitness'].iloc[-1]:.6f}")
    print(f"Improvement: {history['best_fitness'].iloc[0] / history['best_fitness'].iloc[-1]:.1f}x")

def main():
    """Main test runner"""
    print("PSO Parameter Optimization Test Suite")
    print("=====================================")
    
    # Check if required files exist
    if not Path("gfn1-base.toml").exists():
        print("Error: gfn1-base.toml not found. Please ensure base parameter file exists.")
        return
    
    if not Path("h2_ccsd_data.csv").exists():
        print("Warning: h2_ccsd_data.csv not found. Generating reference data first...")
        distances = np.linspace(0.4, 4.0, 500)
        study = H2StudyManager(distances)
        ccsd_config = CalcConfig(method=CalcMethod.CCSD, basis="cc-pVTZ")
        study.add_method("CCSD/cc-pVTZ", ccsd_config, "h2_ccsd_data.csv")
        print("Reference data generated.")
    
    print("\nChoose test to run:")
    print("1. Quick PSO test (10 particles, 20 iterations)")
    print("2. Full PSO optimization (30 particles, 100 iterations)")
    print("3. Compare optimization methods")
    print("4. Analyze PSO convergence")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        success = quick_pso_test()
        if success:
            print("\nQuick test successful! You can now run the full optimization.")
    elif choice == "2":
        full_pso_optimization()
    elif choice == "3":
        compare_optimization_methods()
    elif choice == "4":
        analyze_pso_convergence()
    else:
        print("Invalid choice. Running quick test by default.")
        quick_pso_test()

if __name__ == "__main__":
    main() 
