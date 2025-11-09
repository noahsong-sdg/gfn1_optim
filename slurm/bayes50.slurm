#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bayes_50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --output=bayes50_%j.out
#SBATCH --error=bayes50_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

module purge
module load gcc
module load cmake
module load anaconda

cd /projects/noso3320/work/gfn
conda activate /projects/noso3320/software/gfn_env

# Set OMP_NUM_THREADS=1 to avoid oversubscription when using multiple workers
# Each Optuna worker runs DFTB+ in a separate process
export OMP_NUM_THREADS=1

# Use most CPUs for parallel trials (leave 2 for system overhead)
python src/cli.py bayes big config/gfn1-base.toml --name 50calls_bayes --n-calls 50 --n-jobs 28 --output results/bayes_50
