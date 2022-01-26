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
#source /p/home/jusers/cherepashkin1/juwels/scalable_dl/sdl_venv/sdl_venv/activate.sh
source /p/home/jusers/cherepashkin1/jureca/cherepashkin1/virt_enves/venv0/activate.sh
# make sure all GPUs on a node are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python -u /p/home/jusers/cherepashkin1/jureca/circles/finetune_test/old_mains/main_24.10.py -epoch 1 bn 5 bs 40 lr 1e-3 -no_minmax -no_minmax3dimage -downsample 1 -no_normalize -no_center -no_classicnorm -ampl 441 -cmscrop 550 -rescale 550 -no_use_adasum -expnum "e056" -hidden_dim "hidden_dim = np.hstack((np.repeat(32, 1),1500))" -chidden_dim "chidden_dim = np.hstack((96,128,np.repeat(256, 3)))" -kernel_sizes "kernel_sizes = np.hstack((7,np.repeat(3, 5)))" -inputt "img" -no_feature_extract -weight_decay 0 -num_input_images 1 -specie "598" -loss_between "pc" -no_rand_angle -no_zero_angle -outputt "single_f_n" -csvname "598csv9" -pc_scale 100 -no_view_sep -no_rot_dirs -no_noise_input