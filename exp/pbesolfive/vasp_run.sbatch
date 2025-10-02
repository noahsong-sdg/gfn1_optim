#!/bin/bash

#SBATCH --account=ucb-general 
##SBATCH --account=ucb298_peak3 ##  Group allocation that can be swapped in or out. 

## QoS_type
#SBATCH --qos=normal

## Partition_name
##SBATCH --partition=atesting ##Testing partition for alpine
#SBATCH --partition=amilan ##Partition to actually run some jobs

## Job_name
#SBATCH --job-name=pbesolfive ## edit here to give job name (4-5 letters should be descriptive).

#SBATCH -o o.%j                 # Name of stdout output file (%j expands to jobId)
#SBATCH -e e.%j                 # name of error file

## Number_of_nodes
#SBATCH --nodes=1

## Number_of_processors
#SBATCH --ntasks=32 ## how many cores do you want 

## Job_Wall_Time
#SBATCH --time=1:00:00

#SBATCH --mail-user=noso3320@colorado.edu
##SBATCH --mail-type=BEGIN     # email me when the job starts, commented out
#SBATCH --mail-type=END       # email me when the job finishes, commented out
#SBATCH --mail-type=FAIL        # email me if the job fails

#SBATCH --export=NONE                   # for modules on new nodes

# load modules required for MPI and VASP
module purge; module load intel/2022.1.2 impi/2021.5.0 mkl/2022.0.2 hdf5/1.12.1

# Create ECHO option if not already present
if test "`echo -e`" = "-e" ; then ECHO=echo ; else ECHO="echo -e" ; fi

## useful directories
VASP_DIR=/projects/$USER/software/alpine_vasp_6.4.2/bin/vasp_std
SUBM_DIR=$PWD # submitting directory
SCR_DIR=/scratch/alpine/$USER/${SLURM_JOB_ID}

### make SCR_DIR for this job
mkdir -p -v ${SCR_DIR} ## makes a scratch directory to do the job in.

## Record calculation to jobs_log file
printf '%s\n' "${SLURM_JOB_ID}  ${SLURM_JOB_NAME}  $(date)  $HOSTNAME    ${SBATCH_ACCOUNT}  ${SUBM_DIR}" >> ~/jobs_log

StartTime=$( date +%s.%N ) # bash timing 
## this clean_up function moves output to the submitting directory and
## cleans up scratch space
clean_up () {

    StopTime=$( date +%s.%N )
    printf "Elapsed(sec):  %.3F\n" $( $ECHO  "${StopTime}-${StartTime}"|bc ) >  VASP_${SLURM_JOB_ID}_Timed.bash

    $ECHO
    $ECHO "Moving results to head node ${SUBM_DIR}..."
    mv -v $SCR_DIR/* $SUBM_DIR
    $ECHO "...done"

    wait

    rm -r -v -f ${SCR_DIR}

    $ECHO "... done"
    $ECHO
    $ECHO "Changing directory to ${SUBM_DIR}"
    cd ${SUBM_DIR}

    mv OUTCAR OUTCAR.${SLURM_JOB_ID}
    cp POSCAR POSCAR.${SLURM_JOB_ID}
    cp INCAR INCAR.${SLURM_JOB_ID}
    mv vasprun.xml vasprun.xml.${SLURM_JOB_ID} ##LARGE FILE
    mv CONTCAR POSCAR

    if [ -f PROCAR ]; then
      ## because I can't figure out FORTRAN to save my life...:
      ## this line fixes spacing in the header of the PROCAR file
        sed -i "s/ion      s     py     pz     px    dxy    dyz    dz2   dxz    dx2    tot/ions   py        pz        px       dxy       dyz       dz2       dxz       dx2      tot/g" PROCAR
    fi

    $ECHO "...done"
    $ECHO
    $ECHO "over and out"
}

set -P; shopt -s extglob

cp -v ${SUBM_DIR}/!(o.${SLURM_JOB_ID}|e.${SLURM_JOB_ID}|README) ${SCR_DIR}
rm -v ${SUBM_DIR}/!(o.${SLURM_JOB_ID}|e.${SLURM_JOB_ID}|README)
cd ${SCR_DIR}

$ECHO
$ECHO
$ECHO "running vasp program..."
## run program
export OMP_NUM_THREADS=1
mpirun -np ${SLURM_NTASKS} ${VASP_DIR} > screen.${SLURM_JOB_ID}

$ECHO "vasp done... over and out"
 clean_up

if [ -s POSCAR ]; then
        : # do nothing 
else
        cp POSCAR POSCAR.empty
        cp POSCAR.${SLURM_JOB_ID} POSCAR
fi

## sometimes, jobs aren't recorded into ~/jobs_log; check if this happens and append to
## end of file if so
vijID=$( grep "${SLURM_JOB_ID}" ~/jobs_log | cut -f1 -d' ' )
if [[ $vijID == ${SLURM_JOB_ID} ]]; then
        exit 0
else
        ## Record calculation to jobs_log file
        printf '\%s{\textbackslash}n' "${SLURM_JOB_ID}  ${SLURM_JOB_NAME}  $(date)  $HOSTNAME}
        ${SBATCH_ACCOUNT}  ${SUBM_DIR}" >> ~/jobs_log
fi

