#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=si2_ccsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=si2_ccsd_%j.out
#SBATCH --error=si2_ccsd_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake

cd /projects/noso3320/work/gfn1_optim

# Run the Si2 CCSD data generation with pixi
conda activate /projects/noso3320/software/gfn_env
python src/generate_si2_ccsd.py 
