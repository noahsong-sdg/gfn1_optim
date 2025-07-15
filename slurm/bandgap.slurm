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

# Properly activate conda environment
source /projects/noso3320/software/gfn_env/bin/activate

# Set environment variables for parallel processing
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# PySCF specific optimizations
export PYSCF_MAX_MEMORY=4000
export PYSCF_NUM_THREADS=16

# Run with test mode first to verify everything works
echo "Running test mode with first 3 structures..."
python src/bandgap.py --test --fast --method pbe --output test_band_gaps_fast.csv

echo "Test completed. Check the output for any errors."
