#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=pso_allgap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=pso_%j.out
#SBATCH --error=pso_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn
conda activate /projects/noso3320/software/gfn_env
#conda install -c conda-forge deap -y
# python src/cli.py pso CdS config/gfn1-base.toml --output results/pso

python src/cli.py pso big config/gfn1-base.toml --name pso_20iter_allgap --output-dir results/pso_allgap
