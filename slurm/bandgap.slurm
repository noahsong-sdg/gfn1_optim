#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bandgap_calc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

# Set environment variables for parallel processing
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# Run band gap calculation with PySCF molecular approach
echo "Starting band gap calculation for all structures in trainall.xyz using PySCF molecular"
echo "Job started at: $(date)"

# Run with test mode first to verify everything works
echo "Running test mode with first 3 structures..."
pixi run python src/bandgap.py --test --method pbe --basis sto-3g --output test_band_gaps_molecular.csv

# If test succeeds, run full calculation
if [ $? -eq 0 ]; then
    echo "Test successful. Running full calculation with PySCF PBE/sto-3g molecular..."
    pixi run python src/bandgap.py --xyz_file trainall.xyz --method pbe --basis sto-3g --output trainall_band_gaps_molecular.csv
else
    echo "Test failed. Check error messages above."
    exit 1
fi

echo "Job completed at: $(date)"
