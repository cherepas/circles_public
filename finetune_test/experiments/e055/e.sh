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

srun --cpu-bind=none,v python3 -u /p/home/jusers/cherepashkin1/jureca/circles/finetune_test/main.py -expdescr '3d reconstruction of resampled pc' -bn 150 -epoch 100 -bs 40 -no_classicnorm -no_normalize -lr 1e-3 -expnum e055 -cmscrop 550 -hidden_dim 'hidden_dim = np.hstack((np.repeat(1e4, 4),441)).astype(int)' -loss_between 'f' -criterion 'L1' -inputt 'img' -outputt 'single_f_n' -haf -use_existing_csv -no_rand_angle -num_input_images 1 -no_view_sep -csvname '598csv10' -no_rot_dirs -man_dist