"""
Analysis and visualization tools for genetic algorithm optimization results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import toml
import tempfile
import os

from ga import TBLiteParameterGA, GAConfig
from h2_v2 import MolecularCalculator, DissociationCurveGenerator, CalcConfig, CalcMethod


class GAAnalyzer:
    """Analyze and visualize genetic algorithm optimization results"""
    
    def __init__(self, 
                 ga_fitness_file: str,
                 ga_params_file: str,
                 base_params_file: str = "gfn1-base.toml"):
        """
        Initialize analyzer
        
        Args:
            ga_fitness_file: CSV file with fitness history
            ga_params_file: TOML file with optimized parameters
            base_params_file: Original base parameters
        """
        self.fitness_history = pd.read_csv(ga_fitness_file)
        
        with open(ga_params_file, 'r') as f:
            self.optimized_params = toml.load(f)
            
        with open(base_params_file, 'r') as f:
            self.base_params = toml.load(f)
    
    def plot_fitness_evolution(self, save_path: Optional[str] = None):
        """Plot fitness evolution over generations"""
        plt.figure(figsize=(12, 8))
        
        # Plot best and average fitness
        plt.subplot(2, 1, 1)
        plt.plot(self.fitness_history['generation'], 
                self.fitness_history['best_fitness'], 
                'b-', linewidth=2, label='Best Fitness')
        plt.plot(self.fitness_history['generation'], 
                self.fitness_history['avg_fitness'], 
                'r--', linewidth=1, label='Average Fitness')
        plt.fill_between(self.fitness_history['generation'],
                        self.fitness_history['avg_fitness'] - self.fitness_history['std_fitness'],
                        self.fitness_history['avg_fitness'] + self.fitness_history['std_fitness'],
                        alpha=0.3, color='red', label='±1 std')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution During GA Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot fitness diversity (std)
        plt.subplot(2, 1, 2)
        plt.plot(self.fitness_history['generation'], 
                self.fitness_history['std_fitness'], 
                'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Standard Deviation')
        plt.title('Population Diversity (Fitness Std Dev)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fitness evolution plot saved to {save_path}")
        else:
            plt.show()
    
    def compare_h2_curves(self, save_path: Optional[str] = None, show_train_test_split: bool = True):
        """Compare H2 dissociation curves: original vs GA-optimized vs reference"""
        distances = np.linspace(0.5, 2.5, 50)
        
        # Load reference CCSD data and align it to match the GA's test distances
        if Path("h2_ccsd_data.csv").exists():
            ref_data = pd.read_csv("h2_ccsd_data.csv")
            
            # If reference data has different number of points, interpolate to match
            if len(ref_data) != len(distances):
                from scipy.interpolate import interp1d
                interp_func = interp1d(ref_data['Distance'], ref_data['Energy'], 
                                     kind='cubic', bounds_error=False, fill_value='extrapolate')
                ref_energies = interp_func(distances)
            else:
                ref_energies = ref_data['Energy'].values
        else:
            print("Warning: No CCSD reference data found")
            ref_energies = None
        
        # Calculate with original parameters
        orig_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file="gfn1-base.toml",
            spin=1
        )
        orig_calc = MolecularCalculator(orig_config)
        orig_gen = DissociationCurveGenerator(orig_calc)
        orig_data = orig_gen.generate_h2_curve(distances)
        
        # Calculate with GA-optimized parameters  
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(self.optimized_params, f)
            opt_param_file = f.name
        
        opt_config = CalcConfig(
            method=CalcMethod.XTB_CUSTOM,
            param_file=opt_param_file,
            spin=1
        )
        opt_calc = MolecularCalculator(opt_config)
        opt_gen = DissociationCurveGenerator(opt_calc)
        opt_data = opt_gen.generate_h2_curve(distances)
        
        # Clean up temp file
        os.unlink(opt_param_file)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Convert to relative energies
        if ref_energies is not None:
            ref_relative = ref_energies - np.min(ref_energies)
            plt.plot(distances, ref_relative, 'k-', linewidth=2, 
                    label='CCSD/cc-pVTZ (Reference)', marker='o', markersize=3)
        
        orig_energies = orig_data['Energy'].values
        orig_relative = orig_energies - np.min(orig_energies)
        plt.plot(distances, orig_relative, 'r--', linewidth=2, 
                label='Original GFN1-xTB', marker='s', markersize=3)
        
        opt_energies = opt_data['Energy'].values
        opt_relative = opt_energies - np.min(opt_energies)
        plt.plot(distances, opt_relative, 'b-', linewidth=2, 
                label='GA-Optimized GFN1-xTB', marker='^', markersize=3)
        
        plt.xlabel('H-H Distance (Å)', fontsize=12)
        plt.ylabel('Relative Energy (Hartree)', fontsize=12)
        plt.title('H₂ Dissociation Curves: GA Optimization Results', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Overlay training/test split if requested
        if show_train_test_split:
            try:
                # Recreate the GA object to get train/test split info
                from ga import TBLiteParameterGA, GAConfig
                config = GAConfig(population_size=2, generations=1, max_workers=1)
                ga = TBLiteParameterGA(
                    base_param_file="gfn1-base.toml",
                    config=config,
                    train_fraction=0.8
                )
                
                # Get training and test points
                train_distances = ga.train_distances
                train_energies = ga.train_energies
                test_distances = ga.test_distances  
                test_energies = ga.test_energies
                
                # Convert to relative energies for plotting
                if ref_energies is not None:
                    ref_min = np.min(ref_energies)
                else:
                    ref_min = np.min(train_energies)
                    
                train_rel = train_energies - ref_min
                test_rel = test_energies - ref_min
                
                # Plot training points
                plt.scatter(train_distances, train_rel, c='green', s=50, alpha=0.7, 
                           marker='o', edgecolors='darkgreen', linewidth=1,
                           label='Training Points', zorder=5)
                
                # Plot test points  
                plt.scatter(test_distances, test_rel, c='orange', s=50, alpha=0.7,
                           marker='s', edgecolors='darkorange', linewidth=1, 
                           label='Test Points', zorder=5)
                           
                plt.legend(fontsize=11, loc='upper right')
                
            except Exception as e:
                print(f"Warning: Could not overlay train/test split: {e}")

        # Calculate and display RMSE
        if ref_energies is not None:
            orig_rmse = np.sqrt(np.mean((ref_relative - orig_relative)**2))
            opt_rmse = np.sqrt(np.mean((ref_relative - opt_relative)**2))
            
            # Calculate separate train/test RMSE if split data is available
            rmse_text = f'RMSE vs CCSD:\nOriginal: {orig_rmse:.4f} Ha\nGA-Optimized: {opt_rmse:.4f} Ha\nImprovement: {((orig_rmse - opt_rmse)/orig_rmse)*100:.1f}%'
            
            try:
                # Calculate train/test specific RMSE for GA-optimized
                from scipy.interpolate import interp1d
                
                # Interpolate GA results to train/test points
                interp_opt = interp1d(distances, opt_relative, kind='cubic', bounds_error=False, fill_value='extrapolate')
                
                # Get train/test points (reuse from above if available)
                from ga import TBLiteParameterGA, GAConfig
                config = GAConfig(population_size=2, generations=1, max_workers=1)
                ga = TBLiteParameterGA("gfn1-base.toml", config=config, train_fraction=0.8)
                
                if ref_energies is not None:
                    ref_min = np.min(ref_energies)
                    train_rel = ga.train_energies - ref_min
                    test_rel = ga.test_energies - ref_min
                    
                    opt_train_rel = interp_opt(ga.train_distances)
                    opt_test_rel = interp_opt(ga.test_distances)
                    
                    train_rmse = np.sqrt(np.mean((train_rel - opt_train_rel)**2))
                    test_rmse = np.sqrt(np.mean((test_rel - opt_test_rel)**2))
                    
                    rmse_text += f'\n\nGA Train RMSE: {train_rmse:.4f} Ha\nGA Test RMSE: {test_rmse:.4f} Ha\nTrain/Test Ratio: {train_rmse/test_rmse:.2f}'
                    
            except Exception as e:
                print(f"Warning: Could not calculate train/test RMSE: {e}")
            
            plt.text(0.02, 0.98, rmse_text,
                    transform=plt.gca().transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"H2 curve comparison saved to {save_path}")
        else:
            plt.show()
    
    def analyze_parameter_changes(self, save_path: Optional[str] = None):
        """Analyze which parameters changed the most during optimization"""
        
        # Extract the parameters that were optimized
        ga = TBLiteParameterGA("gfn1-base.toml")  # Just to get parameter bounds
        param_bounds = ga._define_h2_parameter_bounds()
        
        changes = []
        
        for bound in param_bounds:
            # Get original value
            try:
                orig_val = ga._get_parameter_from_dict(self.base_params, bound.name)
                opt_val = ga._get_parameter_from_dict(self.optimized_params, bound.name)
                
                abs_change = opt_val - orig_val
                rel_change = abs_change / orig_val if orig_val != 0 else float('inf')
                
                changes.append({
                    'parameter': bound.name,
                    'original': orig_val,
                    'optimized': opt_val,
                    'abs_change': abs_change,
                    'rel_change': rel_change,
                    'param_range': bound.max_val - bound.min_val,
                    'normalized_change': abs_change / (bound.max_val - bound.min_val)
                })
            except Exception as e:
                print(f"Warning: Could not analyze parameter {bound.name}: {e}")
        
        changes_df = pd.DataFrame(changes)
        changes_df['abs_rel_change'] = np.abs(changes_df['rel_change'])
        changes_df = changes_df.sort_values('abs_rel_change', ascending=False)
        
        # Plot parameter changes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Relative changes
        n_show = min(15, len(changes_df))
        top_changes = changes_df.head(n_show)
        
        colors = ['red' if x > 0 else 'blue' for x in top_changes['rel_change']]
        ax1.barh(range(n_show), top_changes['rel_change'], color=colors, alpha=0.7)
        ax1.set_yticks(range(n_show))
        ax1.set_yticklabels([param.split('.')[-1] for param in top_changes['parameter']], fontsize=10)
        ax1.set_xlabel('Relative Change')
        ax1.set_title('Top Parameter Changes (Relative)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Normalized changes (by parameter range)
        colors = ['red' if x > 0 else 'blue' for x in top_changes['normalized_change']]
        ax2.barh(range(n_show), np.abs(top_changes['normalized_change']), color=colors, alpha=0.7)
        ax2.set_yticks(range(n_show))
        ax2.set_yticklabels([param.split('.')[-1] for param in top_changes['parameter']], fontsize=10)
        ax2.set_xlabel('Normalized Change (by parameter range)')
        ax2.set_title('Top Parameter Changes (Normalized)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter analysis saved to {save_path}")
        else:
            plt.show()
        
        # Print summary
        print("\nTop 10 Parameter Changes:")
        print(changes_df[['parameter', 'original', 'optimized', 'rel_change']].head(10).to_string(index=False))
        
        return changes_df
    
    def convergence_analysis(self, save_path: Optional[str] = None):
        """Analyze convergence characteristics"""
        fitness_vals = self.fitness_history['best_fitness'].values
        
        # Calculate convergence metrics
        final_fitness = fitness_vals[-1]
        max_fitness = np.max(fitness_vals)
        
        # Find generation where 95% of final improvement was achieved
        if len(fitness_vals) > 1:
            total_improvement = final_fitness - fitness_vals[0]
            target_fitness = fitness_vals[0] + 0.95 * total_improvement
            
            converged_gen = np.where(fitness_vals >= target_fitness)[0]
            if len(converged_gen) > 0:
                convergence_gen = converged_gen[0]
            else:
                convergence_gen = len(fitness_vals) - 1
        else:
            convergence_gen = 0
        
        # Plot convergence analysis
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.fitness_history['generation'], fitness_vals, 'b-', linewidth=2)
        plt.axhline(y=target_fitness, color='red', linestyle='--', 
                   label=f'95% convergence (gen {convergence_gen})')
        plt.axvline(x=convergence_gen, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot improvement rate
        plt.subplot(1, 2, 2)
        if len(fitness_vals) > 1:
            improvement_rate = np.diff(fitness_vals)
            plt.plot(self.fitness_history['generation'][1:], improvement_rate, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Improvement per Generation')
        plt.title('Improvement Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence analysis saved to {save_path}")
        else:
            plt.show()
        
        # Print convergence summary
        print(f"\nConvergence Summary:")
        print(f"  Final fitness: {final_fitness:.6f}")
        print(f"  Maximum fitness: {max_fitness:.6f}")
        print(f"  Total improvement: {total_improvement:.6f}")
        print(f"  95% convergence at generation: {convergence_gen}")
        print(f"  Total generations: {len(fitness_vals)}")
    
    def plot_train_test_split(self, save_path: Optional[str] = None):
        """Visualize the train/test split distribution"""
        try:
            # Create GA object to get split info
            from ga import TBLiteParameterGA, GAConfig
            config = GAConfig(population_size=2, generations=1, max_workers=1)
            ga = TBLiteParameterGA(
                base_param_file="gfn1-base.toml",
                config=config,
                train_fraction=0.8
            )
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Distance distribution
            ax1.scatter(ga.train_distances, np.ones(len(ga.train_distances)), 
                       c='green', s=30, alpha=0.7, label='Training Points')
            ax1.scatter(ga.test_distances, np.zeros(len(ga.test_distances)), 
                       c='orange', s=30, alpha=0.7, label='Test Points')
            ax1.set_xlabel('H-H Distance (Å)')
            ax1.set_ylabel('Split')
            ax1.set_title('Train/Test Split Distribution by Distance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['Test', 'Train'])
            
            # Plot 2: Energy vs Distance with train/test highlighted
            ax2.scatter(ga.train_distances, ga.train_energies, 
                       c='green', s=30, alpha=0.7, label='Training Points')
            ax2.scatter(ga.test_distances, ga.test_energies,
                       c='orange', s=30, alpha=0.7, label='Test Points')
            ax2.plot(ga.full_reference_data['Distance'], ga.full_reference_data['Energy'], 
                    'k-', alpha=0.3, linewidth=1, label='Full CCSD Curve')
            ax2.set_xlabel('H-H Distance (Å)')
            ax2.set_ylabel('Dissociation Energy (Hartree)')
            ax2.set_title('Train/Test Points on H₂ Dissociation Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Train/test split visualization saved to {save_path}")
            else:
                plt.show()
                
            # Print split statistics
            print(f"\nTrain/Test Split Statistics:")
            print(f"Training points: {len(ga.train_distances)}")
            print(f"Test points: {len(ga.test_distances)}")
            print(f"Distance coverage - Train: {ga.train_distances.min():.2f}-{ga.train_distances.max():.2f} Å")
            print(f"Distance coverage - Test: {ga.test_distances.min():.2f}-{ga.test_distances.max():.2f} Å")
            
        except Exception as e:
            print(f"Error creating train/test split plot: {e}")
            import traceback
            traceback.print_exc()

    def generate_full_report(self, output_dir: str = "ga_analysis"):
        """Generate a complete analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating GA analysis report in {output_dir}/")
        
        # Generate all plots
        self.plot_fitness_evolution(output_path / "fitness_evolution.png")
        self.compare_h2_curves(output_path / "h2_curve_comparison.png")
        self.plot_train_test_split(output_path / "train_test_split.png")
        
        param_changes = self.analyze_parameter_changes(output_path / "parameter_changes.png")
        param_changes.to_csv(output_path / "parameter_changes.csv", index=False)
        
        self.convergence_analysis(output_path / "convergence_analysis.png")
        
        # Save final fitness statistics
        final_stats = {
            'final_best_fitness': self.fitness_history['best_fitness'].iloc[-1],
            'final_avg_fitness': self.fitness_history['avg_fitness'].iloc[-1],
            'final_std_fitness': self.fitness_history['std_fitness'].iloc[-1],
            'max_fitness_achieved': self.fitness_history['best_fitness'].max(),
            'total_generations': len(self.fitness_history),
        }
        
        stats_df = pd.DataFrame([final_stats])
        stats_df.to_csv(output_path / "final_statistics.csv", index=False)
        
        print(f"Analysis complete! Check {output_dir}/ for all outputs.")


def main():
    """Example usage of GAAnalyzer"""
    
    # Check if GA results exist
    if not Path("ga_fitness_history.csv").exists():
        print("No GA results found. Run the GA optimization first.")
        return
    
    # Analyze results
    analyzer = GAAnalyzer(
        ga_fitness_file="ga_fitness_history.csv",
        ga_params_file="ga_optimized_params.toml"
    )
    
    # Generate full report
    analyzer.generate_full_report()


if __name__ == "__main__":
    main() 
