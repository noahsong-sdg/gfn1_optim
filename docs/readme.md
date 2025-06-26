silicon gfn1-xtb optimized parameters https://arxiv.org/pdf/2109.10416v1

---
# supercell lattice constants experiments
## lattice constants variation with supercell size
both CdS and GaN undergo annoying amounts of variation and fail to converge to the experimental values. reason unknown

time scales exponentially

## scaling lattice parameters
the same problems occur - convergence to the wrong values


# tblite fit
https://tblite.readthedocs.io/en/latest/tutorial/fitting.html

1. get default params
`tblite param --method gfn1 --output gfn1-base.toml`

2. make tbfit.toml
O = {lgam=[false, true]}
Ti = {lgam=[true, false, true]}

3. check for errors using
`pixi run tblite fit --dry-run gfn1-base.toml tbfit.toml`

4. make run file, run.sh

5. debug by uncommenting the debug line and running
export TBLITE_OUT=data.txt
`TBLITE_PAR=./gfn1-base.toml TBLITE_OUT=data.txt pixi run time ./run.sh`

6. run with `pixi run tblite fit -v gfn1-base.toml tbfit.toml`

7. see results in fitpar.toml

# misc

why is this approach better than using an MLIP to compute the same things?
A: memory constraints for MLIPs


## notes on gfn (gemini deep research)
>  state-of-the-art first-principles calculations of charged defects in periodic supercells are not straightforward. They require the application of sophisticated finite-size correction schemes to obtain physically meaningful results. These corrections account for two primary artifacts: the spurious electrostatic interaction between the charged defect and its periodic images, and the need to align the electrostatic potential of the defective supercell with that of the pristine bulk crystal. Widely used methods include the Makov-Payne, Lany-Zunger (LZ), and Freysoldt-Neugebauer-van de Walle (FNV) schemes

> perform a systematic first-principles investigation of how defect properties evolve across a wide compositional range. The objective is to move beyond calculating a single number and instead discover the underlying physical principles and "chemical trends" that govern defect behavior in this complex ternary or quaternary chemical space.

> Similar approaches have been successfully used to understand hydrogen solubility in complex multicomponent carbides , to explain defect formation and doping limits in II-VI semiconductors like CdTe , and to unravel the role of chemical ordering in high-entropy alloys. The underlying concept—that defect interactions and formation energies are strongly dependent on the local composition in an alloy—is well-established

>  use applied lattice strain as a clean, controllable parameter to tune the electronic properties of defects. This project explores the fundamental coupling between the mechanical and electronic degrees of freedom in the crystal, with the practical goal of engineering the material's properties for better device performance.

