"""
Minimal Bayesian Optimization for TBLite parameter tuning
Simplified version for debugging and understanding core concepts
"""

import numpy as np
import pandas as pd
import toml
import tempfile
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real

# Simple paths
PROJECT_ROOT = Path.cwd()
BASE_PARAM_FILE = PROJECT_ROOT / "config" / "gfn1-base.toml" 
REFERENCE_DATA = PROJECT_ROOT / "results" / "curves" / "h2_ccsd_500.csv"

class SimpleBayesOptimizer:
    """Minimal Bayesian optimizer with just 3 parameters for H2"""
    
    def __init__(self):
        print("Setting up simple Bayesian optimizer...")
        
        # Load base parameters once
        with open(BASE_PARAM_FILE, 'r') as f:
            self.base_params = toml.load(f)
        
        # Define just 3 key parameters to optimize
        self.param_names = [
            "hamiltonian.xtb.kpol",      # Polarization parameter
            "element.H.levels[0]",       # H 1s energy level  
            "element.H.gam"              # H gamma parameter
        ]
        
        # Parameter bounds [min, max, default]
        self.param_bounds = {
            "hamiltonian.xtb.kpol": [1.0, 5.0, 2.85],
            "element.H.levels[0]": [-15.0, -8.0, -10.92], 
            "element.H.gam": [0.2, 0.8, 0.47]
        }
        
        # Create optimization dimensions
        self.dimensions = [
            Real(self.param_bounds[name][0], self.param_bounds[name][1], name=name)
            for name in self.param_names
        ]
        
        # Load reference data (use subset for speed)
        ref_data = pd.read_csv(REFERENCE_DATA)
        # Take every 5th point to speed up evaluation
        self.ref_data = ref_data.iloc[::5].copy()
        self.distances = self.ref_data['Distance'].values
        self.ref_energies = self.ref_data['Energy'].values
        
        print(f"Loaded {len(self.distances)} reference points")
        print(f"Optimizing {len(self.param_names)} parameters")
        
        # Tracking
        self.iteration = 0
        self.best_rmse = float('inf')
    
    def set_parameter_value(self, param_dict, path, value):
        """Set parameter using dot notation (simplified)"""
        if path == "hamiltonian.xtb.kpol":
            param_dict["hamiltonian"]["xtb"]["kpol"] = value
        elif path == "element.H.levels[0]":
            param_dict["element"]["H"]["levels"][0] = value
        elif path == "element.H.gam":
            param_dict["element"]["H"]["gam"] = value
    
    def create_temp_params(self, param_values):
        """Create temporary parameter file"""
        # Copy base parameters
        params = self.base_params.copy()
        
        # Apply new parameter values
        for i, name in enumerate(self.param_names):
            self.set_parameter_value(params, name, param_values[i])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(params, f)
            return f.name
    
    def evaluate_params(self, param_values):
        """Evaluate parameter set against reference H2 curve"""
        try:
            self.iteration += 1
            
            # Create parameter file
            param_file = self.create_temp_params(param_values)
            
            # Import calculator (local import to avoid issues)
            from calc import CalcMethod, CalcConfig, GeneralCalculator, DissociationCurveGenerator
            from config import get_system_config
            
            # Set up calculation
            calc_config = CalcConfig(
                method=CalcMethod.XTB_CUSTOM,
                param_file=param_file,
                spin=1
            )
            
            system_config = get_system_config("H2")
            calculator = GeneralCalculator(calc_config, system_config)
            generator = DissociationCurveGenerator(calculator)
            
            # Calculate H2 curve
            calc_data = generator.generate_curve(distances=self.distances, save=False)
            calc_energies = calc_data['Energy'].values
            
            # Compare to reference (relative energies)
            ref_relative = self.ref_energies - np.min(self.ref_energies)
            calc_relative = calc_energies - np.min(calc_energies)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((ref_relative - calc_relative)**2))
            
            # Track best
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                print(f"Evaluation {self.iteration}: New best RMSE = {rmse:.6f}")
                print(f"  Parameters: {dict(zip(self.param_names, param_values))}")
            
            # Clean up
            Path(param_file).unlink(missing_ok=True)
            
            return rmse
            
        except Exception as e:
            print(f"Evaluation {self.iteration} failed: {e}")
            return float('inf')
    
    def optimize(self, n_calls=50):
        """Run Bayesian optimization"""
        print(f"\nStarting Bayesian optimization with {n_calls} evaluations...")
        
        # Start with default parameters
        default_values = [self.param_bounds[name][2] for name in self.param_names]
        print("Evaluating default parameters first...")
        
        # Run optimization
        result = gp_minimize(
            func=self.evaluate_params,
            dimensions=self.dimensions,
            n_calls=n_calls,
            n_initial_points=10,        # Random exploration first
            x0=[default_values],        # Start from defaults
            y0=[self.evaluate_params(default_values)],
            acq_func="EI",             # Expected Improvement
            random_state=42
        )
        
        print(f"\nOptimization complete!")
        print(f"Best RMSE: {result.fun:.6f}")
        print(f"Best parameters:")
        for i, name in enumerate(self.param_names):
            print(f"  {name}: {result.x[i]:.6f}")
        
        # Save best parameters
        output_file = PROJECT_ROOT / "results" / "parameters" / "bayes0_best.toml"
        self.save_best_params(result.x, output_file)
        
        return result
    
    def save_best_params(self, best_values, filename):
        """Save best parameters to file"""
        params = self.base_params.copy()
        
        for i, name in enumerate(self.param_names):
            self.set_parameter_value(params, name, best_values[i])
        
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            toml.dump(params, f)
        
        print(f"Best parameters saved to: {filename}")

def main():
    """Run simple Bayesian optimization"""
    
    # Create optimizer
    optimizer = SimpleBayesOptimizer()
    
    # Run optimization (fewer calls for testing)
    result = optimizer.optimize(n_calls=30)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 
