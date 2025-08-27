#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bandgap_calc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bandgap_%j.out
#SBATCH --error=bandgap_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load intel
module load impi
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn1_optim

conda activate /projects/noso3320/software/gfn_env

# https://gpaw.readthedocs.io/documentation/parallel_runs/parallel_runs.html 
gpaw -P 8 python src/tests/bandgap_gpaw.py

