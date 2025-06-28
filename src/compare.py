#!/usr/bin/env python3
"""
General Method Comparison Script
Compare multiple parameter sets (TOML files) against CCSD reference data

pixi run python src/compare.py --ccsd --pure \
    --params results/parameters/ga_optimized.toml \
        results/parameters/pso_optimized_params.toml \
        results/parameters/bayes_opt.toml \
            --names ga pso bayes--output results/comparison 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toml
from pathlib import Path
from typing import Optional, Dict, List, Union
import argparse
import logging

from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
from config import get_system_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Portable paths
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"

class MethodComparisonAnalyzer:
    """General analyzer for comparing multiple calculation methods"""
    
    def __init__(self, system_name: str = "H2"):
        self.system_name = system_name
        self.system_config = get_system_config(system_name)
        
        # Results storage
        self.results = {}
        self.metrics = {}
        
    def add_method(self, name: str, method_type: str, param_file: Optional[str] = None) -> None:
        """Add a calculation method to compare
        
        Args:
            name: Display name for the method
            method_type: 'ccsd', 'gfn1_pure', or 'custom'
            param_file: Path to TOML parameter file (required for 'custom')
        """
        
        if method_type == 'ccsd':
            # Load CCSD reference data
            ccsd_file = RESULTS_DIR / "curves" / f"{self.system_name.lower()}_ccsd_data.csv"
            if ccsd_file.exists():
                self.results[name] = pd.read_csv(ccsd_file)
                logger.info(f"Loaded CCSD data from {ccsd_file}")
            else:
                logger.warning(f"CCSD data not found at {ccsd_file}")
                
        elif method_type == 'gfn1_pure':
            # Pure GFN1-xTB calculation
            config = CalcConfig(method=CalcMethod.GFN1_XTB)
            self._calculate_curve(name, config)
            
        elif method_type == 'custom':
            # Custom parameters from TOML file
            if not param_file:
                raise ValueError("param_file required for custom method")
            
            param_path = Path(param_file)
            if not param_path.exists():
                raise FileNotFoundError(f"Parameter file not found: {param_file}")
            
            config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=str(param_path)
            )
            self._calculate_curve(name, config)
            
        else:
            raise ValueError(f"Unknown method_type: {method_type}")
    
    def _calculate_curve(self, name: str, config: CalcConfig, distances: Optional[np.ndarray] = None) -> None:
        """Calculate dissociation curve for a given configuration"""
        
        if distances is None:
            distances = np.linspace(0.4, 4.0, 100)
        
        logger.info(f"Calculating curve for {name}...")
        logger.info(f"  Method: {config.method}")
        logger.info(f"  Distance range: {distances[0]:.2f} - {distances[-1]:.2f} Å ({len(distances)} points)")
        
        calculator = GeneralCalculator(config, self.system_config)
        generator = DissociationCurveGenerator(calculator)
        
        curve_data = generator.generate_curve(distances=distances, save=False)
        self.results[name] = curve_data
        
        # Debug info
        energies = curve_data['Energy'].values
        logger.info(f"Completed calculation for {name}")
        logger.info(f"  Energy range: {energies.min():.6f} to {energies.max():.6f} Hartree")
        logger.info(f"  Min energy at distance: {distances[np.argmin(energies)]:.3f} Å")
    
    def calculate_metrics(self, reference_method: str) -> Dict[str, Dict[str, float]]:
        """Calculate RMSE, MAE, and Max Error vs reference method"""
        
        if reference_method not in self.results:
            logger.error(f"Reference method '{reference_method}' not found")
            return {}
        
        ref_data = self.results[reference_method]
        ref_energies = ref_data['Energy'].values
        ref_relative = ref_energies - np.min(ref_energies)
        
        logger.info(f"Calculating metrics vs {reference_method}:")
        
        for method_name, data in self.results.items():
            if method_name == reference_method:
                continue
                
            calc_energies = data['Energy'].values
            calc_relative = calc_energies - np.min(calc_energies)
            
            # Ensure same length (interpolate if needed)
            if len(calc_relative) != len(ref_relative):
                from scipy.interpolate import interp1d
                ref_distances = ref_data['Distance'].values
                calc_distances = data['Distance'].values
                
                interp_func = interp1d(calc_distances, calc_relative, 
                                     kind='cubic', bounds_error=False, fill_value='extrapolate')
                calc_relative = interp_func(ref_distances)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            mae = np.mean(np.abs(ref_relative - calc_relative))
            max_error = np.max(np.abs(ref_relative - calc_relative))
            
            self.metrics[method_name] = {
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error
            }
            
            logger.info(f"  {method_name}:")
            logger.info(f"    RMSE: {rmse:.6f} Hartree")
            logger.info(f"    MAE:  {mae:.6f} Hartree")
            logger.info(f"    Max Error: {max_error:.6f} Hartree")
        
        return self.metrics
    
    def plot_comparison(self, save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """Create comprehensive comparison plot"""
        
        if not self.results:
            logger.error("No results to plot. Add methods first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Define colors and styles - ensuring distinct appearance for each method
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        
        method_styles = {}
        for i, method_name in enumerate(self.results.keys()):
            # Special styling for key methods to avoid overlap
            if 'CCSD' in method_name:
                style = {
                    'color': 'black',
                    'linestyle': '-',
                    'linewidth': 3,
                    'marker': 'o',
                    'markersize': 5,
                    'alpha': 1.0,
                    'markevery': 5  # Show markers every 5 points
                }
            elif 'Pure' in method_name or 'pure' in method_name or 'GFN1-xTB (Pure)' in method_name:
                style = {
                    'color': 'red',
                    'linestyle': '--',
                    'linewidth': 2.5,
                    'marker': 's',
                    'markersize': 4,
                    'alpha': 0.9,
                    'markevery': 7
                }
            elif 'PSO' in method_name:
                style = {
                    'color': 'blue',
                    'linestyle': '-.',
                    'linewidth': 2,
                    'marker': '^',
                    'markersize': 4,
                    'alpha': 0.9,
                    'markevery': 6
                }
            elif 'GA' in method_name:
                style = {
                    'color': 'green',
                    'linestyle': ':',
                    'linewidth': 2,
                    'marker': 'D',
                    'markersize': 4,
                    'alpha': 0.9,
                    'markevery': 8
                }
            else:
                # Fallback for other methods
                style = {
                    'color': colors[i % len(colors)],
                    'linestyle': linestyles[i % len(linestyles)],
                    'linewidth': 2,
                    'marker': markers[i % len(markers)],
                    'markersize': 3,
                    'alpha': 0.8,
                    'markevery': 9
                }
            
            method_styles[method_name] = style
        
        # Plot 1: Full dissociation curves
        for method_name, data in self.results.items():
            distances = data['Distance'].values
            energies = data['Energy'].values
            # For dissociation curves, zero should be at the furthest distance (dissociation limit)
            dissociation_energy = energies[-1]  # Energy at largest distance
            relative_energies = energies - dissociation_energy
            
            style = method_styles[method_name]
            ax1.plot(distances, relative_energies, label=method_name, **style)
        
        ax1.set_xlabel('Distance (Å)', fontsize=12)
        ax1.set_ylabel('Relative Energy (Hartree)', fontsize=12)
        ax1.set_title(f'{self.system_name} Dissociation Curves Comparison', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xlim(0.4, 4.0)
        
        # Plot 2: Zoom in on equilibrium region
        for method_name, data in self.results.items():
            distances = data['Distance'].values
            energies = data['Energy'].values
            # For dissociation curves, zero should be at the furthest distance (dissociation limit)
            dissociation_energy = energies[-1]  # Energy at largest distance
            relative_energies = energies - dissociation_energy
            
            # Focus on equilibrium region
            eq_mask = (distances >= 0.5) & (distances <= 2.0)
            
            style = method_styles[method_name]
            ax2.plot(distances[eq_mask], relative_energies[eq_mask], 
                    label=method_name, **style)
        
        ax2.set_xlabel('Distance (Å)', fontsize=12)
        ax2.set_ylabel('Relative Energy (Hartree)', fontsize=12)
        ax2.set_title('Equilibrium Region (Zoomed)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_xlim(0.5, 2.0)
        
        # Add metrics text box if available
        if self.metrics:
            metrics_text = "RMSE vs Reference (Hartree):\n"
            for method, metric in self.metrics.items():
                metrics_text += f"{method}: {metric['rmse']:.4f}\n"
            
            ax1.text(0.02, 0.98, metrics_text.strip(),
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add hyperparameter information
        hyperparams_text = self._get_hyperparameter_info()
        if hyperparams_text:
            ax2.text(0.98, 0.02, hyperparams_text,
                    transform=ax2.transAxes, fontsize=8,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self, reference_method: str, output_file: Optional[str] = None) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("=" * 70)
        report.append(f"METHOD COMPARISON REPORT - {self.system_name}")
        report.append("=" * 70)
        report.append("")
        
        # Methods included
        report.append("METHODS COMPARED:")
        report.append("-" * 20)
        for i, method_name in enumerate(self.results.keys(), 1):
            report.append(f"{i}. {method_name}")
        report.append("")
        
        # Performance metrics
        if self.metrics:
            report.append(f"PERFORMANCE METRICS vs {reference_method}:")
            report.append("-" * 40)
            
            # Sort by RMSE
            sorted_methods = sorted(self.metrics.items(), key=lambda x: x[1]['rmse'])
            
            for method, metrics in sorted_methods:
                report.append(f"{method}:")
                report.append(f"  RMSE:      {metrics['rmse']:.6f} Hartree")
                report.append(f"  MAE:       {metrics['mae']:.6f} Hartree") 
                report.append(f"  Max Error: {metrics['max_error']:.6f} Hartree")
                report.append("")
            
            # Ranking
            report.append("RANKING BY RMSE (Best to Worst):")
            report.append("-" * 35)
            for i, (method, metrics) in enumerate(sorted_methods, 1):
                report.append(f"{i}. {method:<30} {metrics['rmse']:.6f} Hartree")
            report.append("")
        
        # Equilibrium analysis
        report.append("EQUILIBRIUM BOND DISTANCES:")
        report.append("-" * 30)
        for method_name, data in self.results.items():
            energies = data['Energy'].values
            distances = data['Distance'].values
            min_idx = np.argmin(energies)
            eq_distance = distances[min_idx]
            min_energy = energies[min_idx]
            
            report.append(f"{method_name:<30} {eq_distance:.3f} Å ({min_energy:.6f} Ha)")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text

    def _get_hyperparameter_info(self) -> str:
        """Get hyperparameter information for optimization methods"""
        hyperparams = []
        
        # Check for PSO results
        if any("PSO" in name for name in self.results.keys()):
            hyperparams.append("PSO Hyperparameters:")
            hyperparams.append("• Particles: 12")
            hyperparams.append("• Iterations: 25") 
            hyperparams.append("• Inertia: 0.5-0.9 (adaptive)")
            hyperparams.append("• Cognitive: 1.5")
            hyperparams.append("• Social: 1.5")
            hyperparams.append("")
        
        # Check for GA results
        if any("GA" in name for name in self.results.keys()):
            hyperparams.append("GA Hyperparameters:")
            hyperparams.append("• Population: 20")
            hyperparams.append("• Generations: 30")
            hyperparams.append("• Mutation Rate: 0.1")
            hyperparams.append("• Crossover Rate: 0.8")
            hyperparams.append("• Workers: 8")
            hyperparams.append("")
        
        return "\n".join(hyperparams).strip()


def main():
    """Command line interface for method comparison"""
    
    parser = argparse.ArgumentParser(description='Compare multiple calculation methods')
    parser.add_argument('--system', default='H2', help='System name (default: H2)')
    parser.add_argument('--ccsd', action='store_true', help='Include CCSD reference')
    parser.add_argument('--pure', action='store_true', help='Include pure GFN1-xTB')
    parser.add_argument('--params', nargs='*', help='TOML parameter files to compare')
    parser.add_argument('--names', nargs='*', help='Display names for parameter files')
    parser.add_argument('--reference', default='CCSD', help='Reference method for metrics')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MethodComparisonAnalyzer(args.system)
    
    # Add methods
    if args.ccsd:
        analyzer.add_method('CCSD/cc-pVTZ', 'ccsd')
    
    if args.pure:
        analyzer.add_method('GFN1-xTB (Pure)', 'gfn1_pure')
    
    if args.params:
        param_names = args.names if args.names else [f"Custom-{i+1}" for i in range(len(args.params))]
        
        for param_file, name in zip(args.params, param_names):
            try:
                analyzer.add_method(name, 'custom', param_file)
            except Exception as e:
                logger.error(f"Failed to add {param_file}: {e}")
    
    # Auto-detect common parameter files if none specified
    if not args.params and not args.ccsd and not args.pure:
        logger.info("No methods specified. Auto-detecting available parameter files...")
        
        # Add CCSD if available
        ccsd_file = RESULTS_DIR / "curves" / f"{args.system.lower()}_ccsd_data.csv"
        if ccsd_file.exists():
            analyzer.add_method('CCSD/cc-pVTZ', 'ccsd')
        
        # Add pure GFN1-xTB
        analyzer.add_method('GFN1-xTB (Pure)', 'gfn1_pure')
        
        # Look for optimization results
        param_files = {
            'PSO-Optimized': RESULTS_DIR / "parameters" / "pso_optimized_params.toml",
            'GA-Optimized': RESULTS_DIR / f"{args.system}" / "ga_optimized.toml",
        }
        
        for name, param_file in param_files.items():
            if param_file.exists():
                analyzer.add_method(name, 'custom', str(param_file))
    
    if not analyzer.results:
        logger.error("No methods to compare. Specify --ccsd, --pure, or --params")
        return
    
    # Calculate metrics
    reference_methods = [name for name in analyzer.results.keys() if args.reference.lower() in name.lower()]
    if reference_methods:
        reference_method = reference_methods[0]
        analyzer.calculate_metrics(reference_method)
    else:
        reference_method = list(analyzer.results.keys())[0]
        logger.warning(f"Reference method '{args.reference}' not found. Using '{reference_method}'")
        if len(analyzer.results) > 1:
            analyzer.calculate_metrics(reference_method)
    
    # Generate outputs
    output_dir = Path(args.output) if args.output else RESULTS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    if not args.no_plot:
        plot_path = output_dir / f"{args.system}_method_comparison_v2.png"
        analyzer.plot_comparison(save_path=str(plot_path), show_plot=False)
    
    # Report
    report_path = output_dir / f"{args.system}_comparison_report.txt"
    report = analyzer.generate_report(reference_method, str(report_path))
    
    # Print summary
    print("\n" + report)
    print(f"\nFiles generated:")
    if not args.no_plot:
        print(f"  Plot: {plot_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main() 
