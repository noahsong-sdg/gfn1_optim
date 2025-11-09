#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=fast12cma2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=cma2_%j.out
#SBATCH --error=cma2_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn
conda activate /projects/noso3320/software/gfn_env
# python src/cli.py cma2 CdS config/gfn1-base.toml --output results/cma2 
python src/cli.py cma2 big config/gfn1-base.toml --name cma2_100iter_allgap --output-dir results/cma2_fast12
