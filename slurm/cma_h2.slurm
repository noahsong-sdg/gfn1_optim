#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=cma_h2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=cma_h2_%j.out
#SBATCH --error=cma_h2_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn1_optim
conda activate /projects/noso3320/software/gfn_env
conda install -c conda-forge cmaes
python src/cma.py H2 
