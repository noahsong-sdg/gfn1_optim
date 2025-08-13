#!/usr/bin/env python3
"""
band gap for dataset creation
"""
import os, argparse, time
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
import pyscf
import pyscf.pbc.gto
import pyscf.pbc.dft
from pyscf.pbc import dft as pbc_dft
os.environ['OMP_NUM_THREADS'] = "16" 
logger = logging.getLogger(__name__)

# learn about gth-dzvp
def atoms_to_pyscf_cell(atoms: Atoms, basis: str = 'gth-dzvp') -> pyscf.pbc.gto.Cell:
    cell = pyscf.pbc.gto.Cell()
    cell.a = atoms.get_cell()
    cell.unit = 'Angstrom'
    cell.basis = basis
    cell.pseudo = 'gth-lda'
    
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        cell.atom.append([symbol, pos.tolist()])
    
    cell.build()
    logger.info(cell)
    return cell

def calculate_bandgap(atoms: Atoms, method: str = 'pbe', basis: str = 'sto-3g') -> Dict[str, float]:
    cell = atoms_to_pyscf_cell(atoms, basis)    
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
    nocc = nelec // 2  # 

    vb_list = []
    cb_list = []
    direct_gaps = []
    logger.info(e_kn)
    # this stuff irrelevant for now bc just have k pt
    for en in e_kn:
        occ = en[:nocc]
        unocc = en[nocc:]
        if len(occ) > 0:
            vb_list.append(np.max(occ))
        if len(unocc) > 0:
            cb_list.append(np.min(unocc))
        if len(occ) > 0 and len(unocc) > 0:
            direct_gaps.append(np.min(unocc) - np.max(occ))
    if vb_list and cb_list:
        global_vbm = np.max(vb_list)
        global_cbm = np.min(cb_list)
        indirect_gap = global_cbm - global_vbm
        direct_gap = np.min(direct_gaps) if direct_gaps else indirect_gap
        # 
        return e_kn
    """{
            'indirect_gap': indirect_gap,
            'direct_gap': direct_gap,
            'global_vbm': global_vbm,
            'global_cbm': global_cbm,
            'method': method,
            'basis': basis,
            'converged': mf.converged,
            'total_energy': mf.e_tot,
            'error': None
        }"""

        

def process_structures(xyz_file: str, output_file: str = None, method: str = 'pbe', 
                     basis: str = 'sto-3g') -> pd.DataFrame:
    structures = list(ase.io.read(xyz_file, index=':'))
    
    results = []
    start_time = time.time()
    
    for i, atoms in enumerate(structures):
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
        result = calculate_bandgap(atoms, method, basis)
        calc_time = time.time() - calc_start
        
        results.append({**structure_info, **result, 'calculation_time': calc_time})
        
        # Save every 10 structures
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(f"temp_results_{i+1}.csv", index=False)
    
    # Create final DataFrame and save
    df = pd.DataFrame(results)
    if output_file is None:
        output_file = f"band_gap_results_{Path(xyz_file).stem}_{method}_{basis}.csv"
    
    df.to_csv(output_file, index=False)
    
    # Print summary
    successful = df[df['converged'] == True]
    
    if len(successful) > 0:
        gaps = successful['direct_gap'].values
        logger.info(f"Band gap range: {np.min(gaps):.3f} - {np.max(gaps):.3f} eV")
        logger.info(f"Mean: {np.mean(gaps):.3f} ± {np.std(gaps):.3f} eV")
    
    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Results saved in {output_file}")
    
    return df

def main():
    """Main function to run band gap calculations"""
    parser = argparse.ArgumentParser(description='Calculate band gaps using pyscf')
    parser.add_argument('--xyz_file', default='trainall.xyz')
    parser.add_argument('--output', default=None)
    parser.add_argument('--method', default='pbe', choices=['pbe', 'pbe0', 'b3lyp'])
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("testing first three structures")
        structures = list(ase.io.read(args.xyz_file, index=':3'))
        test_file = 'test_structures.xyz'
        ase.io.write(test_file, structures)
        df = process_structures(test_file, args.output, args.method, args.basis)
        os.remove(test_file)
    else:
        df = process_structures(args.xyz_file, args.output, args.method, args.basis)
    
    return df

if __name__ == "__main__":
    main() 
