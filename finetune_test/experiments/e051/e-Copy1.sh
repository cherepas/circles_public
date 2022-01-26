#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --output=./output/sh3d-%j.out
#SBATCH --error=./output/sh3d-%j.err
#SBATCH --time=23:00:00
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

srun --cpu-bind=none,v python3 -u /p/home/jusers/cherepashkin1/jureca/circles/finetune_test/sh.py -expdescr 'cmscrop, fix image rotation angle, classic norm, L1 norm' -bn 5 -epoch 500 -bs 40 -classicnorm -lr 5e-7 -expnum e051 -cmscrop 550 -hidden_dim 'hidden_dim = np.array([5000,2500,1000,441])' -loss_between 'f' -criterion 'L1' -inputt 'img'