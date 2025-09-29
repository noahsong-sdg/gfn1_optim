#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=gad_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=gad_%j.out
#SBATCH --error=gad_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn
conda activate /projects/noso3320/software/gfn_env
# python src/cli.py gad CdS config/gfn1-base.toml --output results/gad
python src/clipy pso big config/gfn1-base.toml --output-dir results/gad
