#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bayes0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=bayes0_%j.out
#SBATCH --error=bayes0_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn1_optim

# Activate conda environment
conda activate /projects/noso3320/software/gfn_env

# Run the simple version
python src/bayes0.py 
