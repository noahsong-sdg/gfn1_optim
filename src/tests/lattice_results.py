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
    
def main():
    pso_calc = TBLiteASECalculator(
        param_file="../results/parameters/CdS_pso.toml",
        method="gfn1",
        electronic_temperature=300.0,
        charge=0.0,
        spin=0,
    )
    ga_calc = TBLiteASECalculator(
        param_file="../results/parameters/CdS_ga.toml",
        method="gfn1",
        electronic_temperature=300.0,
        charge=0.0,
        spin=0,
    )
    bayes_calc = TBLiteASECalculator(
        param_file="../results/bayes/CdS_bayes.toml",
        method="gfn1",
        electronic_temperature=300.0,
        charge=0.0,
        spin=0,
    )

    default_df = get_params(TBLite(method="GFN1-xTB", electronic_temperature=300.0))
    print("Default")
    print(default_df)
    pso_df = get_params(pso_calc)
    print("PSO")
    print(pso_df)
    ga_df = get_params(ga_calc)
    print("GA")
    print(ga_df)
    bayes_df = get_params(bayes_calc)
    print("Bayes")
    print(bayes_df)    
    # DEBUG: Concatenate all DataFrames with method name as first column

    # Add a column to each DataFrame to indicate the method name
    default_df['Method'] = 'Default'
    pso_df['Method'] = 'PSO'
    ga_df['Method'] = 'GA'
    bayes_df['Method'] = 'Bayes'

    # Reorder columns so 'Method' is first
    def reorder_columns(df):
        cols = df.columns.tolist()
        if 'Method' in cols:
            cols.insert(0, cols.pop(cols.index('Method')))
        return df[cols]

    default_df = reorder_columns(default_df)
    pso_df = reorder_columns(pso_df)
    ga_df = reorder_columns(ga_df)
    bayes_df = reorder_columns(bayes_df)

    # Concatenate all DataFrames
    all_results_df = pd.concat([default_df, pso_df, ga_df, bayes_df], ignore_index=True)
    all_results_df.to_csv("lattice_results.csv", index=False)
    print(all_results_df)
if __name__ == "__main__":
    main()

