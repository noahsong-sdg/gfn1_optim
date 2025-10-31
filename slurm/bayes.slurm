#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bayes_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --output=bayes_%j.out
#SBATCH --error=bayes_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn
conda activate /projects/noso3320/software/gfn_env
# python src/cli.py bayes CdS config/gfn1-base.toml --output results/bayes 
#python src/cli.py bayes big config/gfn1-base.toml --output-dir results/bayes
conda install -n base mamba
mamba install 'dftbplus=*=mpi_openmpi_*'
python src/cli.py bayes big config/gfn1-base.toml --n-calls 200 --n-jobs 12 --output results/bayes
