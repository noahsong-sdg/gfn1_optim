#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --job-name=bandgap_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=bandgap_gpu_%j.out
#SBATCH --error=bandgap_gpu_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=noso3320@colorado.edu

# Load required modules for AMD GPU
module purge
module load gcc
module load cmake
module load anaconda
module load rocm/5.4.3  # AMD ROCm for GPU acceleration

# Set up environment
cd /projects/noso3320/work/gfn1_optim
conda activate /projects/noso3320/software/gfn_env

# Set environment variables for AMD GPU
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# Install CuPy for AMD GPU if not already installed
# pip install cupy-cuda12x  # For CUDA 12.x
# pip install cupy-rocm-5-4  # For AMD ROCm 5.4

echo "Starting GPU-accelerated band gap calculations..."
echo "GPU: $(rocm-smi --showproductname | grep 'Product Name')"
echo "Memory: $(rocm-smi --showmemory | grep 'Memory')"

# Run with test mode first to verify GPU acceleration works
echo "Running test mode with first 5 structures..."
python src/bandgap_gpu.py --test --method pbe --basis def2-svp --output test_band_gaps_gpu.csv --workers 4

# If test succeeds, run full calculation
if [ $? -eq 0 ]; then
    echo "Test successful! Running full calculation..."
    python src/bandgap_gpu.py --method pbe --basis def2-svp --output band_gaps_gpu_full.csv --workers 4
else
    echo "Test failed! Check GPU setup and PySCF installation."
    exit 1
fi

echo "GPU-accelerated band gap calculations completed!" 
