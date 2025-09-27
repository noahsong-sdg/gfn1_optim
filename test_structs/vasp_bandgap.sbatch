#!/bin/bash

#SBATCH --account=ucb-general
#SBATCH --qos=normal
#SBATCH --partition=amilan

#SBATCH --job-name=CdS_BGaps        
#SBATCH --array=0-199             

#SBATCH -o o.%A_%a               
#SBATCH -e e.%A_%a                

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00

#SBATCH --mail-user=noso3320@colorado.edu
#SBATCH --mail-type=FAIL            
#SBATCH --mail-type=END          

module purge
module load intel/2022.1.2 impi/2021.5.0 mkl/2022.0.2 hdf5/1.12.1
VASP_DIR=/projects/$USER/software/alpine_vasp_6.4.2/bin/vasp_std

# 1. Determine the directory for this specific array task
#    This creates names like "structure_000", "structure_001", etc.
STRUCTURE_DIR=$(printf "structure_%03d" ${SLURM_ARRAY_TASK_ID})

# 2. Check if the directory exists
if [ ! -d "$STRUCTURE_DIR" ]; then
  echo "Error: Directory ${STRUCTURE_DIR} not found!"
  exit 1
fi

# 3. Navigate into the structure's directory
cd ${STRUCTURE_DIR}

# 4. Copy the generic input files from the parent directory
#    The POSCAR should already be here.
cp ../INCAR .
cp ../POTCAR .
cp ../KPOINTS .

echo "Starting VASP in ${PWD} for array task ${SLURM_ARRAY_TASK_ID}"

# 5. Run VASP
#    Output files (OUTCAR, vasprun.xml, etc.) will be written in this directory.
export OMP_NUM_THREADS=1
mpirun -np ${SLURM_NTASKS} ${VASP_DIR} > vasp_screen.out

echo "VASP done in ${PWD}. Task ${SLURM_ARRAY_TASK_ID} finished."

exit 0
