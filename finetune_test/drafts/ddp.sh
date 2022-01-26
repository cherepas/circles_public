#!/bin/bash
# bash -x for detailed shell debugging

# We choose in this solution example $K$ = 8 workers to run
# Total number of tasks $T$ is equal to number of workers $T=K=8$

# As each worker gets one GPU, we should allocate $K=8$ GPUs in total.

# If we take $g=4$ GPUs per each node,
# this then corresponds to $N=K/g=8/4=2$ compute nodes to allocate,

# Tasks per node is equal to number of local GPUs to be allocated, $t=g=4$

# For cpu cores, we have then $c=C/g=96/4 = 24$. Total number of available cores
# $C$ on Booster is  2 * 24 * 2 = 96 threads.



#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=./output/sh3d-%j.out
#SBATCH --error=./output/sh3d-%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4

#SBATCH --account=delia-mp

#source /p/project/training2107/course2021_working_environment/activate.sh
module load GCC/9.3.0 OpenMPI/4.1.0rc1 Python/3.8.5 Horovod/0.20.3-Python-3.8.5 PyTorch/1.7.0-Python-3.8.5 scikit/2020-Python-3.8.5 torchvision/0.8.2-Python-3.8.5 
#OpenMPI/4.1.0rc1 OpenMPI mpi4py TensorFlow 
HOROVOD_MPI_THREADS_DISABLE=0
source /p/home/jusers/cherepashkin1/juwels/scalable_dl/sdl_venv/sdl_venv/activate.sh
# make sure all GPUs on a node are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# use srun to start Horovod code
srun --cpu-bind=none,v python3 -u ddp.py 
