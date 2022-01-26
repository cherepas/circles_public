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
ml load PyTorch
#OpenMPI/4.1.0rc1 OpenMPI mpi4py TensorFlow
HOROVOD_MPI_THREADS_DISABLE=0
#source /p/home/jusers/cherepashkin1/juwels/scalable_dl/sdl_venv/sdl_venv/activate.sh
source /p/home/jusers/cherepashkin1/juwels/scalable_dl/sdl_venv/sdl_venv/activate.sh
#source /p/home/jusers/cherepashkin1/jureca/cherepashkin1/virt_enves/venv0/activate.sh
# make sure all GPUs on a node are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"
WANDB_MODE=offline
#WANDB_MODE=disabled
python -u /p/home/jusers/cherepashkin1/jureca/circles/finetune_test/main.py -expdescr "truly normalized dataset" -bn 1000 -epoch 300 -bs 50 -lr 1e-2 -expnum e054 -cmscrop 550 -hidden_dim "hidden_dim = np.hstack((np.repeat(32, 1),1500))" -lb "pc" -criterion "L1" -inputt "img" -outputt "single_f_n" -num_input_images 1 -view_sep -csvname "598csv11" -no_rot_dirs -machine "jureca" -parallel "horovod" -wandb "" 
