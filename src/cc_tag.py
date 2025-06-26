from pyscf import gto, scf, cc
import numpy as np
import pandas as pd
import os
from pyscf.grad import ccsd as ccsd_grad

def calc_h2_cc(distance):
    mol = gto.Mole()
    mol.atom = f"""
    H 0 0 0
    H 0 0 {distance}
    """
    mol.basis = "cc-pVTZ"
    mol.symmetry = False
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()
    #cc_scanner = cc.CCSD(scf.UHF(mol)).nuc_grad_method().as_scanner()
    #e_total, grad = cc_scanner(mol)
    #mycc = cc_grad.base
    # Run CCSD calculation
    ccsd = cc.CCSD(mf)
    e_corr_uccsd, t1, t2 = ccsd.kernel()

    # The TOTAL UCCSD energy is the sum of the UHF energy and the correlation energy.
    e_total = mf.e_tot + e_corr_uccsd
    ae = [e_total/2, e_total/2]
    grad = ccsd.nuc_grad_method().kernel()
    return e_total, ae, grad

def make_coordFile(distance, coord_file: str):
    distance_bohr = distance * 1.88973
    with open(coord_file, "w") as f:
        f.write("$coord\n")
        f.write(f" 0.0 0.0 0.0 h\n") 
        f.write(f" 0.0 0.0 {distance_bohr:.10f} h\n")   
        f.write("$end\n")

def make_tagFile(distance, output_file: str):
    energy, ae, grad = calc_h2_cc(distance)
    virial = np.zeros((3, 3)) # since our system is an isolated molecule

    with open(output_file, "w") as f:
        f.write("energy             :real:0:\n")
        f.write(f" {energy:23.16E}\n")

        f.write("energies                 :real:0:\n")
        f.write(f" {ae[0]:23.16E} {ae[1]:23.16E}\n")

        f.write("gradient           :real:2:3,2\n")
        for i in range(2):  # 2 atoms
            f.write(f" {grad[i,0]:23.16E} {grad[i,1]:23.16E} {grad[i,2]:23.16E}\n")
        
        f.write("virial             :real:2:3,3\n")
        for i in range(3):
            f.write(f" {virial[i,0]:23.16E} {virial[i,1]:23.16E} {virial[i,2]:23.16E}\n")

def trainset(distances, base_dir="data"):
    for i, dist in enumerate(distances):
        dir = f"h2_{i:03d}_{dist:.2f}"
        full_dir = os.path.join(base_dir, dir)

        os.makedirs(full_dir, exist_ok=True)

        coord_file = os.path.join(full_dir, "coord")
        make_coordFile(dist, coord_file)

        tagfile = os.path.join(full_dir, "reference.tag")
        make_tagFile(dist, tagfile)

if __name__ == "__main__":
    # Create reference.tag for H2 at 0.74 Angstrom (equilibrium distance)
    distances = np.linspace(0.1, 2.0, 100)
    trainset(distances, "data")
