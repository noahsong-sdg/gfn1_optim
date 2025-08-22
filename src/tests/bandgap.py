#!/usr/bin/env python3
"""
band gap for dataset creation

https://pyscf.org/user/pbc/scf.html
recommends using multi-grid algorithms or range separation integral algorithms to 
"reduce cost of Fock build"

# of plane waves used is controlled by KE cutoff, Cell.ke_cutoff
gaussian density fitting is more economical

GTH doesnt have cadnium sadge
i could do GaN but they dont have Zn 

mom can i have gth for Cd
we have gth at home
gth for Cd at home:
raise BasisNotFoundError(f'Basis for {symb} not found in {basisfile}')

basis from https://github.com/cp2k/cp2k/blob/master/data/BASIS_MOLOPT
pp from https://github.com/cp2k/cp2k/blob/master/data/GTH_POTENTIALS
returned
Very diffused basis functions are found in the basis set. They may lead to severe
  linear dependence and numerical instability.  You can set  cell.exp_to_discard=0.1
  to remove the diffused Gaussians whose exponents are less than 0.1.

  WARN: Even tempered Gaussians are generated as DF auxbasis for  S Cd

  WARN: memory usage of outcore_auxe2 may be 0.35 times over max_memory
"""
import os, argparse, time
import logging
from typing import Dict
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
import pyscf
import pyscf.pbc.gto
import pyscf.pbc.dft
from pyscf.pbc import dft as pbc_dft

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress PySCF warnings about JSON serialization
logging.getLogger('pyscf').setLevel(logging.WARNING)
os.environ['OMP_NUM_THREADS'] = "16" 
logger = logging.getLogger(__name__)

def atoms_to_pyscf_cell(atoms: Atoms) -> pyscf.pbc.gto.Cell:
    cell = pyscf.pbc.gto.Cell()
    cell.a = atoms.get_cell()
    cell.unit = 'Angstrom'
    cell.basis = {'Cd': pyscf.gto.basis.parse_cp2k.parse("""
Cd DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q12
 1
 2 0 3 5 2 2 2 1
      2.617301292227  0.056942862843  0.085229465042  0.032277317911  0.051510843904 -0.270406154064 -0.064920306950 -0.038770557595
      1.315617684053 -0.111910427538 -0.202708060768 -0.096634842893 -0.059783928261 -0.407839310946 -0.152553100915 -0.110107626346
      0.573375899853 -0.420173785813  1.542506765787 -0.083107933622  0.615455610766 -0.363233448773  0.032121836723 -0.034014284171
      0.222222213763  0.564376116618 -2.827046600082  0.419549254990 -1.767302700397 -0.168907403589 -0.117391708980 -0.107254575233
      0.076307136525  0.789967059806  1.744522478935  0.795089873672  1.158998093237 -0.013172843723  1.045519107194  1.059683456620                                             
    """),
                'S': pyscf.gto.basis.parse_cp2k.parse("""
S  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q6
 1
 2 0 2 4 2 2 1
      2.215854692813  0.170962878400 -0.080726543800  0.092191824200  0.057845138800  0.113762894700
      1.131470525271  0.127069405600 -0.209877313900 -0.162197093800 -0.094737441500  0.350414093700
      0.410168143974 -0.733925381700  0.683497090800 -0.605594737600 -0.369172638100  0.866785684700
      0.140587330023 -0.176971633900 -0.625512739500 -0.213309789800  1.155699504700  0.217880463100
""")}
    # cell.pseudo = 'gth-lda'
    cell.pseudo = {'Cd': pyscf.gto.basis.parse_cp2k_pp.parse("""
    #PSEUDOPOTENTIAL
    Cd GTH-PBE-q12 GTH-PBE
    2    0   10
     0.55000000    1     3.63395858
    3
     0.49127900    3    10.11138228    -6.50695409     1.80457679
                                       11.33558598    -4.65939724
                                                       3.69828191
     0.59970999    2     4.00148729    -1.88393578
                                        2.22910288
     0.37787256    2    -6.13703223     1.53571055
                                       -1.74133209
    """), 
    'S': pyscf.gto.basis.parse_cp2k_pp.parse("""
    S GTH-PADE-q6 GTH-LDA-q6 GTH-PADE GTH-LDA
    2    4
     0.42000000    1    -6.55449184
    2
     0.36175665    2     7.90530250    -1.73188130
                                        4.47169830
     0.40528502    1     3.86657900
                                                 """)}
    
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        cell.atom.append([symbol, pos.tolist()])
    
    cell.build()
    logger.info(cell)
    return cell

def calculate_bandgap(atoms: Atoms, method: str = 'pbe') -> Dict[str, float]:
    cell = atoms_to_pyscf_cell(atoms)    
    kpts = np.array([[0, 0, 0]])
    
    # llm tells me that rks is standard for semiconductors, fine for band gaps
    # UKS is needed for magnetic materials and open shelled systems (2x expensive)
    mf = pbc_dft.RKS(cell, kpts)
    # pbe underestimates band gaps by a lot
    if method.lower() == 'pbe':
        mf.xc = 'pbe'
    elif method.lower() == 'pbe0':
        mf.xc = 'pbe0'
        mf.exxdiv = 'ewald' #alts are vcut, gdf, and none
    elif method.lower() == 'b3lyp':
        mf.xc = 'b3lyp'
        mf.exxdiv = 'ewald'
    else:
        mf.xc = method
        
    mf = mf.density_fit()
    mf.conv_tol = 1e-6
    mf.kernel()

    # For a full band structure, you would use a k-point path. Here, just gamma:
    band_kpts = kpts
    e_kn, kpts_bands = mf.get_bands(band_kpts)
    # e_kn shape: (n_kpts, n_bands)
    # n_kpts, n_bands = e_kn.shape
    nelec = cell.nelectron

    logger.info(e_kn)
    
    # Calculate band gap from energy bands
    # For gamma point calculation, we have only one k-point
    energies = e_kn[0]  # Shape: (n_bands,)
    
    # Find HOMO and LUMO
    # HOMO is the highest occupied orbital (index nelec//2 - 1 for closed shell)
    # LUMO is the lowest unoccupied orbital (index nelec//2)
    homo_idx = nelec // 2 - 1
    lumo_idx = nelec // 2
    
    if homo_idx >= 0 and lumo_idx < len(energies):
        homo_energy = energies[homo_idx]
        lumo_energy = energies[lumo_idx]
        band_gap = lumo_energy - homo_energy
        converged = True
    else:
        homo_energy = lumo_energy = band_gap = float('nan')
        converged = False
    
    return {
        'homo_energy': float(homo_energy),
        'lumo_energy': float(lumo_energy), 
        'direct_gap': float(band_gap),
        'converged': converged,
        'n_bands': len(energies),
        'n_electrons': nelec
    }

def process_structures(xyz_file: str, output_file: str = None, method: str = 'pbe') -> pd.DataFrame:
    structures = list(ase.io.read(xyz_file, index=':'))
    
    print(f"\n=== BAND GAP CALCULATION ===")
    print(f"Input file: {xyz_file}")
    print(f"Number of structures: {len(structures)}")
    print(f"Method: {method.upper()}")
    print(f"Output file: {output_file if output_file else 'None (results only printed)'}")
    print(f"Temporary files: temp_results_*.csv (every 10 structures)")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    for i, atoms in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}: {atoms.get_chemical_formula()}")
        logger.info(f"Structure {i+1}/{len(structures)}: {atoms.get_chemical_formula()}")
        
        # Get structure info
        structure_info = {
            'structure_id': i,
            'formula': atoms.get_chemical_formula(),
            'n_atoms': len(atoms),
            'volume': atoms.get_volume(),
            'cell_a': atoms.cell[0, 0] if atoms.cell.any() else 0.0,
            'cell_b': atoms.cell[1, 1] if atoms.cell.any() else 0.0,
            'cell_c': atoms.cell[2, 2] if atoms.cell.any() else 0.0,
        }
        
        # Calculate band gap
        calc_start = time.time()
        try:
            result = calculate_bandgap(atoms, method)
            calc_time = time.time() - calc_start
            print(f"  Band gap: {result['direct_gap']:.3f} eV (converged: {result['converged']})")
            
            results.append({**structure_info, **result, 'calculation_time': calc_time})
        except Exception as e:
            calc_time = time.time() - calc_start
            print(f"  ERROR: {e}")
            logger.error(f"Failed to calculate band gap for structure {i+1}: {e}")
            # Add failed result
            failed_result = {
                'homo_energy': float('nan'),
                'lumo_energy': float('nan'),
                'direct_gap': float('nan'),
                'converged': False,
                'n_bands': 0,
                'n_electrons': 0
            }
            results.append({**structure_info, **failed_result, 'calculation_time': calc_time})

        if (i + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"temp_results_{i+1}.csv", index=False)
            print(f"  Saved temporary results to temp_results_{i+1}.csv")
    
    # Create final DataFrame and save
    df = pd.DataFrame(results)    
    if output_file:
        df.to_csv(output_file, index=False)
    
    # Filter successful calculations
    successful = df[df['converged'] == True]
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Total structures processed: {len(df)}")
    print(f"Successful calculations: {len(successful)}")
    print(f"Failed calculations: {len(df) - len(successful)}")
    
    if len(successful) > 0:
        gaps = successful['direct_gap'].values
        # Filter out NaN values
        valid_gaps = gaps[~np.isnan(gaps)]
        if len(valid_gaps) > 0:
            mean_gap = np.mean(valid_gaps)
            std_gap = np.std(valid_gaps)
            print(f"Mean band gap: {mean_gap:.3f} ± {std_gap:.3f} eV")
            print(f"Min band gap: {np.min(valid_gaps):.3f} eV")
            print(f"Max band gap: {np.max(valid_gaps):.3f} eV")
            logger.info(f"Mean band gap: {mean_gap:.3f} ± {std_gap:.3f} eV")
        else:
            print("No valid band gaps calculated")
            logger.info("No valid band gaps calculated")
    else:
        print("No successful calculations")
        logger.info("No successful calculations")
        
    total_time = time.time() - start_time
    print(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    
    if output_file:
        print(f"Results saved to: {output_file}")
        logger.info(f"Results saved in {output_file}")
    
    return df

def main():
    """Main function to run band gap calculations"""
    parser = argparse.ArgumentParser(description='Calculate band gaps using pyscf')
    parser.add_argument('--xyz_file', default='trainall.xyz')
    parser.add_argument('--output', default=None)
    parser.add_argument('--method', default='pbe', choices=['pbe', 'pbe0', 'b3lyp'])
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("testing first three structures")
        structures = list(ase.io.read(args.xyz_file, index=':3'))
        test_file = 'test_structures.xyz'
        ase.io.write(test_file, structures)
        df = process_structures(test_file, args.output, args.method)
        os.remove(test_file)
    else:
        df = process_structures(args.xyz_file, args.output, args.method)
    
    return df

if __name__ == "__main__":
    main() 
