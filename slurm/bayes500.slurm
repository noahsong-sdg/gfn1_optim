#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bayes_500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --output=bayes500_%j.out
#SBATCH --error=bayes500_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn
conda activate /projects/noso3320/software/gfn_env
python src/cli.py bayes big config/gfn1-base.toml --name 500calls_bayes_big --n-calls 500 --n-jobs 12 --output results/bayes_500
