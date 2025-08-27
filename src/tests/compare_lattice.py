import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
print(sys.path)
from calculators.tblite_ase_calculator import TBLiteASECalculator
from ase.build import bulk
from tblite.ase import TBLite
from ase.optimize import BFGS
#from ase.constraints import UnitCellFilter
from ase.filters import UnitCellFilter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ASE imports for crystal structure visualization
try:
    import ase
    from ase import Atoms
    from ase.build import bulk
    from ase.visualize.plot import plot_atoms
    from ase.geometry.analysis import Analysis
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE not available. Crystal structure plotting disabled.")

# pixi run python tests/lattice_results.py


def get_params(calculator):
    atoms = bulk("CdS", "wurtzite", a = 4.17, c = 6.78)
    atoms.calc = calculator
    ucf = UnitCellFilter(atoms)
    opt = BFGS(ucf)
    opt.run(fmax=0.01)  
    df = pd.DataFrame({
            'a': [atoms.cell.cellpar()[0]],
            'b': [atoms.cell.cellpar()[1]],
            'c': [atoms.cell.cellpar()[2]],
            'alpha': [atoms.cell.cellpar()[3]],
            'beta': [atoms.cell.cellpar()[4]],
            'gamma': [atoms.cell.cellpar()[5]],
            'Energy': [atoms.get_potential_energy() * 0.0367493],  # Convert to Hartree
            #'bandgap': self.get_bandgap(),
            #'elastic': self.getElastic()
        })
    return df

def plot_wurtzite_cds(lattice_params: dict, method_name: str, save_dir: Path = Path("results")) -> None:
    """
    Plot wurtzite CdS crystal structure using lattice parameters
    
    Args:
        lattice_params: Dictionary with keys 'a', 'b', 'c', 'alpha', 'beta', 'gamma'
        method_name: Name of the method for file naming
        save_dir: Base directory to save plots
    """
    
    if not ASE_AVAILABLE:
        print("Warning: ASE not available. Cannot plot crystal structures.")
        return
    
    # Create output directory
    output_dir = save_dir / "crystal_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract lattice parameters
        a = lattice_params.get('a', 4.17)  # Default CdS wurtzite a
        c = lattice_params.get('c', 6.78)  # Default CdS wurtzite c
        
        # Create wurtzite CdS structure
        cd_s = bulk('CdS', 'wurtzite', a=a, c=c)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get atomic positions
        positions = cd_s.get_positions()
        symbols = cd_s.get_chemical_symbols()
        
        # Color mapping
        colors = {'Cd': 'blue', 'S': 'yellow'}
        sizes = {'Cd': 200, 'S': 150}
        
        for pos, symbol in zip(positions, symbols):
            ax.scatter(pos[0], pos[1], pos[2], 
                      c=colors.get(symbol, 'gray'),
                      s=sizes.get(symbol, 100),
                      alpha=0.8,
                      label=symbol)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Wurtzite CdS Structure - {method_name}\n'
                    f'a={a:.3f}Å, c={c:.3f}Å')
        
        # Add legend (only unique symbols)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Save plot
        safe_name = method_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        plot_path = output_dir / f"wurtzite_cds_{safe_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Wurtzite CdS 3D plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error plotting wurtzite structure for {method_name}: {e}")
        import traceback
        traceback.print_exc()

def generate_crystal_plots(results_df: pd.DataFrame, save_dir: Path = Path("results")) -> None:
    """Generate crystal structure plots for all methods in the results DataFrame"""
    
    if not ASE_AVAILABLE:
        print("Warning: ASE not available. Cannot generate crystal plots.")
        return
    
    print("Generating crystal structure plots for all methods...")
    
    for _, row in results_df.iterrows():
        try:
            method_name = row['Method']
            
            # Extract lattice parameters
            lattice_params = {
                'a': row['a'],
                'b': row['b'],
                'c': row['c'],
                'alpha': row['alpha'],
                'beta': row['beta'],
                'gamma': row['gamma'],
            }
            
            plot_wurtzite_cds(lattice_params, method_name, save_dir)
            
        except Exception as e:
            print(f"Failed to generate crystal plot for {method_name}: {e}")

def main():
    pso_calc = TBLiteASECalculator(
        param_file="../results/pso/CdS_pso.toml",
        method="gfn1",
        electronic_temperature=400.0,
        charge=0.0,
        spin=0,
    )
    ga_calc = TBLiteASECalculator(
        param_file="../results/ga/CdS_ga.toml",
        method="gfn1",
        electronic_temperature=400.0,
        charge=0.0,
        spin=0,
    )
    bayes_calc = TBLiteASECalculator(
        param_file="../results/bulk/CdS_bayes.toml",
        method="gfn1",
        electronic_temperature=400.0,
        charge=0.0,
        spin=0,
    )
    cma_calc = TBLiteASECalculator(
        param_file="../results/cma1/CdS_cma1.toml",
        method="gfn1",
        electronic_temperature=400.0,
        charge=0.0,
        spin=0,
    )

    default_df = get_params(TBLite(method="GFN1-xTB", electronic_temperature=400.0))
    print("Default")
    print(default_df)
    bayes_df = get_params(bayes_calc)
    print("Bayes")
    print(bayes_df)    
    cma1_df = get_params(cma_calc)
    print("cma1")
    print(cma1_df) 
    # DEBUG: Concatenate all DataFrames with method name as first column

    # Add a column to each DataFrame to indicate the method name
    default_df['Method'] = 'Default'
    #pso_df['Method'] = 'PSO'
    # ga_df['Method'] = 'GA'
    bayes_df['Method'] = 'Bayes'
    cma1_df['Method'] = 'CMA1'

    # Reorder columns so 'Method' is first
    def reorder_columns(df):
        cols = df.columns.tolist()
        if 'Method' in cols:
            cols.insert(0, cols.pop(cols.index('Method')))
        return df[cols]

    default_df = reorder_columns(default_df)
    #pso_df = reorder_columns(pso_df)
    #ga_df = reorder_columns(ga_df)
    bayes_df = reorder_columns(bayes_df)
    cma1_df = reorder_columns(cma1_df)

    # Concatenate all DataFrames
    all_results_df = pd.concat([default_df, bayes_df, cma1_df], ignore_index=True)
    all_results_df.to_csv("lattice_results.csv", index=False)
    print(all_results_df)
    
    # Generate crystal structure plots for all methods
    generate_crystal_plots(all_results_df, Path("results"))
    
    print(f"\nCrystal plots saved to: results/crystal_plots/")
if __name__ == "__main__":
    main()

