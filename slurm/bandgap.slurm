#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bandgap_calc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=bandgap_%j.out
#SBATCH --error=bandgap_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

# Load required modules
module purge
module load gcc
module load cmake
module load anaconda

# Set up environment
cd /projects/noso3320/work/gfn1_optim

conda activate /projects/noso3320/software/gfn_env

PYTHONPATH=src python -m tests.bandgap --test --method pbe --output test_band_gaps_fast.csv
