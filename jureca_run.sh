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
# module load GCC/9.3.0 OpenMPI/4.1.0rc1 Python/3.8.5 Horovod/0.20.3-Python-3.8.5 PyTorch/1.7.0-Python-3.8.5 scikit/2020-Python-3.8.5 torchvision/0.8.2-Python-3.8.5
# ml load PyTorch
# #OpenMPI/4.1.0rc1 OpenMPI mpi4py TensorFlow
HOROVOD_MPI_THREADS_DISABLE=0
source /p/home/jusers/cherepashkin1/jureca/cherepashkin1/virt_enves/venv1/activate.sh
# # make sure all GPUs on a node are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python -u "main.py" -datapath "/p/project/delia-mp/cherepashkin1/phenoseed/" -epoch 1 -bs 5 -num_input_images 1 -framelim 40 -criterion 'L2' -localexp "" -lr 1e-4 -expnum "e000" -hidden_dim 9 -inputt "img" -outputt "orient" -lb "orient" -no_loadh5 -minmax_fn "" -parallel "torch" -machine "lenovo" -merging "color" -aug_gt "orient" -updateFraction 0.25 -steplr 1000 1 -print_minibatch 1
