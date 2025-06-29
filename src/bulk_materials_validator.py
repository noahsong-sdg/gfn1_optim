"""
Bulk Materials Properties Validator

Computes aggregate loss between reference and computed databases for:
- Ground state energies (formation energies)
- Band gaps
- Elastic constants (bulk modulus, shear modulus)
- Lattice constants (a, b, c, alpha, beta, gamma)

Usage:
    validator = BulkMaterialsValidator("reference_data.csv", "computed_data.csv")
    results = validator.compute_aggregate_loss()
    validator.generate_report("validation_report.html")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy import stats
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MaterialProperty:
    """Container for material property with metadata"""
    name: str
    unit: str
    weight: float = 1.0
    tolerance: float = 0.05  # 5% tolerance by default
    relative_error: bool = True  # Use relative vs absolute error

@dataclass
class ValidationResults:
    """Results from materials validation"""
    total_loss: float
    property_losses: Dict[str, float]
    property_metrics: Dict[str, Dict[str, float]]
    matched_materials: int
    total_materials: int
    correlation_coefficients: Dict[str, float]

class BulkMaterialsValidator:
    """
    Validates bulk materials properties against reference database
    """
    
    def __init__(self, 
                 reference_file: Union[str, Path],
                 computed_file: Union[str, Path],
                 weights_config: Optional[Dict[str, float]] = None):
        """
        Initialize validator with reference and computed data
        
        Args:
            reference_file: Path to reference database (CSV format)
            computed_file: Path to computed database (CSV format)
            weights_config: Dictionary of property weights for loss function
        """
        
        self.reference_file = Path(reference_file)
        self.computed_file = Path(computed_file)
        
        # Define material properties with default weights
        self.properties = {
            'formation_energy': MaterialProperty('Formation Energy', 'eV/atom', weight=2.0),
            'band_gap': MaterialProperty('Band Gap', 'eV', weight=1.5),
            'bulk_modulus': MaterialProperty('Bulk Modulus', 'GPa', weight=1.0),
            'shear_modulus': MaterialProperty('Shear Modulus', 'GPa', weight=1.0),
            'lattice_a': MaterialProperty('Lattice a', 'Å', weight=1.0),
            'lattice_b': MaterialProperty('Lattice b', 'Å', weight=1.0),
            'lattice_c': MaterialProperty('Lattice c', 'Å', weight=1.0),
            'lattice_alpha': MaterialProperty('Lattice α', '°', weight=0.5),
            'lattice_beta': MaterialProperty('Lattice β', '°', weight=0.5),
            'lattice_gamma': MaterialProperty('Lattice γ', '°', weight=0.5),
        }
        
        # Update weights if provided
        if weights_config:
            for prop_name, weight in weights_config.items():
                if prop_name in self.properties:
                    self.properties[prop_name].weight = weight
        
        # Load data
        self.reference_data = self._load_reference_data()
        self.computed_data = self._load_computed_data()
        self.matched_data = self._match_materials()
        
        logger.info(f"Loaded {len(self.reference_data)} reference materials")
        logger.info(f"Loaded {len(self.computed_data)} computed materials")
        logger.info(f"Matched {len(self.matched_data)} materials for validation")
        
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference database"""
        if not self.reference_file.exists():
            raise FileNotFoundError(f"Reference file not found: {self.reference_file}")
        
        df = pd.read_csv(self.reference_file)
        required_columns = ['composition', 'space_group']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in reference data: {missing_columns}")
        
        logger.info(f"Reference data columns: {list(df.columns)}")
        return df
    
    def _load_computed_data(self) -> pd.DataFrame:
        """Load computed database"""
        if not self.computed_file.exists():
            raise FileNotFoundError(f"Computed file not found: {self.computed_file}")
        
        df = pd.read_csv(self.computed_file)
        required_columns = ['composition', 'space_group']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in computed data: {missing_columns}")
        
        logger.info(f"Computed data columns: {list(df.columns)}")
        return df
    
    def _match_materials(self) -> pd.DataFrame:
        """Match materials between reference and computed databases"""
        # Merge on composition and space group
        matched = self.reference_data.merge(
            self.computed_data, 
            on=['composition', 'space_group'], 
            suffixes=('_ref', '_calc'),
            how='inner'
        )
        
        if len(matched) == 0:
            logger.warning("No materials matched between databases!")
            logger.info("Reference compositions (first 10):")
            logger.info(self.reference_data['composition'].head(10).tolist())
            logger.info("Computed compositions (first 10):")
            logger.info(self.computed_data['composition'].head(10).tolist())
        
        return matched
    
    def compute_property_error(self, prop_name: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute error metrics for a specific property
        
        Args:
            prop_name: Name of property to evaluate
            
        Returns:
            errors: Array of errors for each material
            metrics: Dictionary of error metrics (MAE, RMSE, etc.)
        """
        
        if prop_name not in self.properties:
            raise ValueError(f"Unknown property: {prop_name}")
        
        ref_col = f"{prop_name}_ref"
        calc_col = f"{prop_name}_calc"
        
        if ref_col not in self.matched_data.columns or calc_col not in self.matched_data.columns:
            logger.warning(f"Property {prop_name} not found in matched data")
            return np.array([]), {}
        
        # Get valid data (no NaN values)
        mask = (~pd.isna(self.matched_data[ref_col])) & (~pd.isna(self.matched_data[calc_col]))
        ref_values = self.matched_data.loc[mask, ref_col].values
        calc_values = self.matched_data.loc[mask, calc_col].values
        
        if len(ref_values) == 0:
            logger.warning(f"No valid data for property {prop_name}")
            return np.array([]), {}
        
        # Compute errors
        prop = self.properties[prop_name]
        if prop.relative_error:
            # Relative error (avoid division by zero)
            errors = np.where(np.abs(ref_values) > 1e-10, 
                            (calc_values - ref_values) / np.abs(ref_values),
                            calc_values - ref_values)
        else:
            # Absolute error
            errors = calc_values - ref_values
        
        # Compute metrics
        abs_errors = np.abs(errors)
        metrics = {
            'MAE': np.mean(abs_errors),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'Max_Error': np.max(abs_errors),
            'Std_Error': np.std(errors),
            'Mean_Error': np.mean(errors),  # Bias
            'R2': stats.pearsonr(ref_values, calc_values)[0]**2 if len(ref_values) > 1 else 0.0,
            'Count': len(ref_values)
        }
        
        return errors, metrics
    
    def compute_aggregate_loss(self) -> ValidationResults:
        """
        Compute weighted aggregate loss across all properties
        
        Returns:
            ValidationResults object with detailed metrics
        """
        
        property_losses = {}
        property_metrics = {}
        correlation_coeffs = {}
        total_weighted_loss = 0.0
        total_weight = 0.0
        
        for prop_name, prop in self.properties.items():
            errors, metrics = self.compute_property_error(prop_name)
            
            if len(errors) > 0:
                # Use RMSE as the primary loss metric
                property_loss = metrics['RMSE']
                property_losses[prop_name] = property_loss
                property_metrics[prop_name] = metrics
                
                # Add to weighted total
                total_weighted_loss += property_loss * prop.weight
                total_weight += prop.weight
                
                # Store correlation coefficient
                correlation_coeffs[prop_name] = np.sqrt(metrics['R2']) if metrics['R2'] >= 0 else 0.0
                
                logger.info(f"{prop_name}: RMSE = {property_loss:.4f} {prop.unit}, "
                          f"MAE = {metrics['MAE']:.4f}, R² = {metrics['R2']:.3f}, "
                          f"Count = {metrics['Count']}")
            else:
                logger.warning(f"No data available for {prop_name}")
        
        # Compute total loss
        total_loss = total_weighted_loss / total_weight if total_weight > 0 else float('inf')
        
        results = ValidationResults(
            total_loss=total_loss,
            property_losses=property_losses,
            property_metrics=property_metrics,
            matched_materials=len(self.matched_data),
            total_materials=len(self.reference_data),
            correlation_coefficients=correlation_coeffs
        )
        
        logger.info(f"Aggregate Loss: {total_loss:.4f}")
        logger.info(f"Matched {results.matched_materials}/{results.total_materials} materials")
        
        return results
    
    def plot_property_comparison(self, prop_name: str, save_path: Optional[Path] = None) -> None:
        """Plot reference vs computed values for a property"""
        
        ref_col = f"{prop_name}_ref"
        calc_col = f"{prop_name}_calc"
        
        if ref_col not in self.matched_data.columns or calc_col not in self.matched_data.columns:
            logger.warning(f"Cannot plot {prop_name}: data not available")
            return
        
        # Get valid data
        mask = (~pd.isna(self.matched_data[ref_col])) & (~pd.isna(self.matched_data[calc_col]))
        ref_values = self.matched_data.loc[mask, ref_col].values
        calc_values = self.matched_data.loc[mask, calc_col].values
        
        if len(ref_values) == 0:
            logger.warning(f"No valid data to plot for {prop_name}")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax1.scatter(ref_values, calc_values, alpha=0.7, s=50)
        
        # Perfect correlation line
        min_val = min(np.min(ref_values), np.min(calc_values))
        max_val = max(np.max(ref_values), np.max(calc_values))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
        
        ax1.set_xlabel(f'Reference {self.properties[prop_name].name} ({self.properties[prop_name].unit})')
        ax1.set_ylabel(f'Computed {self.properties[prop_name].name} ({self.properties[prop_name].unit})')
        ax1.set_title(f'{self.properties[prop_name].name} Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error histogram
        errors = calc_values - ref_values
        ax2.hist(errors, bins=20, alpha=0.7, density=True)
        ax2.axvline(0, color='r', linestyle='--', alpha=0.8, label='Zero error')
        ax2.set_xlabel(f'Error ({self.properties[prop_name].unit})')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def generate_plots(self, output_dir: Path = Path("validation_plots")) -> None:
        """Generate comparison plots for all properties"""
        
        output_dir.mkdir(exist_ok=True)
        
        for prop_name in self.properties.keys():
            plot_path = output_dir / f"{prop_name}_comparison.png"
            self.plot_property_comparison(prop_name, plot_path)
    
    def generate_report(self, output_path: Path = Path("validation_report.html")) -> None:
        """Generate comprehensive HTML validation report"""
        
        results = self.compute_aggregate_loss()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bulk Materials Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .good {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .error {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Bulk Materials Validation Report</h1>
                <p>Generated for databases: {self.reference_file.name} vs {self.computed_file.name}</p>
                <p>Validation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">Total Aggregate Loss: <strong>{results.total_loss:.4f}</strong></div>
                <div class="metric">Materials Matched: <strong>{results.matched_materials}/{results.total_materials}</strong> 
                    ({100*results.matched_materials/results.total_materials:.1f}%)</div>
            </div>
            
            <div class="section">
                <h2>Property-by-Property Results</h2>
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Unit</th>
                        <th>Weight</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>R²</th>
                        <th>Count</th>
                        <th>Status</th>
                    </tr>
        """
        
        for prop_name, prop in self.properties.items():
            if prop_name in results.property_metrics:
                metrics = results.property_metrics[prop_name]
                r2 = metrics['R2']
                rmse = results.property_losses[prop_name]
                
                # Determine status
                if r2 > 0.9 and rmse < prop.tolerance:
                    status = '<span class="good">Excellent</span>'
                elif r2 > 0.7 and rmse < 2*prop.tolerance:
                    status = '<span class="warning">Good</span>'
                else:
                    status = '<span class="error">Poor</span>'
                
                html_content += f"""
                    <tr>
                        <td>{prop.name}</td>
                        <td>{prop.unit}</td>
                        <td>{prop.weight:.1f}</td>
                        <td>{rmse:.4f}</td>
                        <td>{metrics['MAE']:.4f}</td>
                        <td>{r2:.3f}</td>
                        <td>{metrics['Count']}</td>
                        <td>{status}</td>
                    </tr>
                """
            else:
                html_content += f"""
                    <tr>
                        <td>{prop.name}</td>
                        <td>{prop.unit}</td>
                        <td>{prop.weight:.1f}</td>
                        <td colspan="5"><span class="error">No Data</span></td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Add recommendations based on results
        poor_properties = [name for name, metrics in results.property_metrics.items() 
                          if metrics['R2'] < 0.7]
        
        if poor_properties:
            html_content += f"<li><strong>Poor correlations detected</strong> for: {', '.join(poor_properties)}. Consider parameter optimization.</li>"
        
        if results.matched_materials < 0.5 * results.total_materials:
            html_content += "<li><strong>Low material matching rate</strong>. Check composition and space group naming consistency.</li>"
        
        high_error_props = [name for name, loss in results.property_losses.items() 
                           if loss > 2 * self.properties[name].tolerance]
        
        if high_error_props:
            html_content += f"<li><strong>High errors detected</strong> for: {', '.join(high_error_props)}. Review calculation parameters.</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Write report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Validation report saved to: {output_path}")
    
    def save_results_json(self, output_path: Path = Path("validation_results.json")) -> None:
        """Save validation results to JSON file"""
        
        results = self.compute_aggregate_loss()
        
        results_dict = {
            'total_loss': results.total_loss,
            'matched_materials': results.matched_materials,
            'total_materials': results.total_materials,
            'match_rate': results.matched_materials / results.total_materials,
            'properties': {}
        }
        
        for prop_name, metrics in results.property_metrics.items():
            results_dict['properties'][prop_name] = {
                'unit': self.properties[prop_name].unit,
                'weight': self.properties[prop_name].weight,
                'loss': results.property_losses[prop_name],
                'metrics': metrics,
                'correlation': results.correlation_coefficients.get(prop_name, 0.0)
            }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")

def create_example_data():
    """Create example reference and computed databases for testing"""
    
    # Example compositions (common binary and ternary compounds)
    compositions = [
        'NaCl', 'CsCl', 'LiF', 'MgO', 'CaO', 'TiO2', 'ZnO', 'GaN', 'AlN',
        'SiC', 'BN', 'Al2O3', 'Fe2O3', 'CaTiO3', 'BaTiO3', 'SrTiO3',
        'LaAlO3', 'YAlO3', 'ZrO2', 'HfO2'
    ]
    
    space_groups = ['Fm3m', 'Pm3m', 'P63mc', 'P4mm', 'Pnma', 'R3c', 'Ia3d', 'Pn3m', 'Fd3m', 'I4/mcm']
    
    # Generate reference data (simulate experimental/DFT values)
    np.random.seed(42)
    n_materials = len(compositions)
    
    reference_data = {
        'composition': compositions,
        'space_group': np.random.choice(space_groups, n_materials),
        'formation_energy_ref': np.random.normal(-2.0, 1.5, n_materials),  # eV/atom
        'band_gap_ref': np.random.lognormal(0.5, 0.8, n_materials),  # eV
        'bulk_modulus_ref': np.random.normal(150, 50, n_materials),  # GPa
        'shear_modulus_ref': np.random.normal(80, 30, n_materials),  # GPa
        'lattice_a_ref': np.random.normal(5.0, 1.0, n_materials),  # Å
        'lattice_b_ref': np.random.normal(5.0, 1.0, n_materials),  # Å
        'lattice_c_ref': np.random.normal(5.0, 1.0, n_materials),  # Å
        'lattice_alpha_ref': np.random.normal(90, 5, n_materials),  # °
        'lattice_beta_ref': np.random.normal(90, 5, n_materials),  # °
        'lattice_gamma_ref': np.random.normal(90, 5, n_materials),  # °
    }
    
    # Generate computed data (simulate GFN-xTB calculations with some error)
    computed_data = {
        'composition': compositions,
        'space_group': reference_data['space_group'],  # Same space groups
        'formation_energy_calc': reference_data['formation_energy_ref'] + np.random.normal(0, 0.3, n_materials),
        'band_gap_calc': reference_data['band_gap_ref'] * np.random.normal(1.0, 0.2, n_materials),
        'bulk_modulus_calc': reference_data['bulk_modulus_ref'] + np.random.normal(0, 20, n_materials),
        'shear_modulus_calc': reference_data['shear_modulus_ref'] + np.random.normal(0, 15, n_materials),
        'lattice_a_calc': reference_data['lattice_a_ref'] + np.random.normal(0, 0.1, n_materials),
        'lattice_b_calc': reference_data['lattice_b_ref'] + np.random.normal(0, 0.1, n_materials),
        'lattice_c_calc': reference_data['lattice_c_ref'] + np.random.normal(0, 0.1, n_materials),
        'lattice_alpha_calc': reference_data['lattice_alpha_ref'] + np.random.normal(0, 2, n_materials),
        'lattice_beta_calc': reference_data['lattice_beta_ref'] + np.random.normal(0, 2, n_materials),
        'lattice_gamma_calc': reference_data['lattice_gamma_ref'] + np.random.normal(0, 2, n_materials),
    }
    
    # Save to CSV files
    pd.DataFrame(reference_data).to_csv('example_reference_materials.csv', index=False)
    pd.DataFrame(computed_data).to_csv('example_computed_materials.csv', index=False)
    
    print("Created example databases:")
    print("- example_reference_materials.csv")
    print("- example_computed_materials.csv")

def main():
    """Example usage of BulkMaterialsValidator"""
    
    # Create example data if it doesn't exist
    if not Path('example_reference_materials.csv').exists():
        create_example_data()
    
    # Custom property weights (optional)
    weights = {
        'formation_energy': 3.0,  # Most important for thermodynamics
        'band_gap': 2.0,          # Important for electronic properties
        'bulk_modulus': 1.5,      # Mechanical properties
        'lattice_a': 1.0,         # Structural properties
    }
    
    # Initialize validator
    validator = BulkMaterialsValidator(
        reference_file='example_reference_materials.csv',
        computed_file='example_computed_materials.csv',
        weights_config=weights
    )
    
    # Compute validation results
    results = validator.compute_aggregate_loss()
    
    # Generate detailed outputs
    validator.generate_plots()
    validator.generate_report()
    validator.save_results_json()
    
    print(f"\nValidation complete!")
    print(f"Aggregate Loss: {results.total_loss:.4f}")
    print(f"Materials Matched: {results.matched_materials}/{results.total_materials}")
    print("Check 'validation_report.html' for detailed results")

if __name__ == "__main__":
    main() 
