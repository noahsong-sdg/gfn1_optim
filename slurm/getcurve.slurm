#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=getcurve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=getcurve_%j.out
#SBATCH --error=getcurve_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

# Generate reference data using the general calc.py script
# Examples:
#   Si2 CCSD:  python src/calc.py --system Si2 --method CCSD --output results/curves/si2_ccsd_data.csv
#   H2 CCSD:   python src/calc.py --system H2 --method CCSD --output results/curves/h2_ccsd_data.csv
#   Si2 GFN1:  python src/calc.py --system Si2 --method GFN1-xTB --output results/curves/si2_gfn1_data.csv

# Default: Generate Si2 CCSD data (change as needed)
cd /projects/noso3320/work/gfn1_optim
conda activate /projects/noso3320/software/gfn_env
python src/calc.py --system Si2 --method CCSD --output results/curves/si2_ccsd_data.csv 
