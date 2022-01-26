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
# HOROVOD_MPI_THREADS_DISABLE=0
# #source /p/home/jusers/cherepashkin1/juwels/scalable_dl/sdl_venv/sdl_venv/activate.sh
# #source /p/home/jusers/cherepashkin1/juwels/scalable_dl/sdl_venv/sdl_venv/activate.sh
# source /p/home/jusers/cherepashkin1/jureca/cherepashkin1/virt_enves/venv0/activate.sh
# # make sure all GPUs on a node are visible
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# python -u /p/home/jusers/cherepashkin1/jureca/circles/finetune_test/old_mains/main_02.11.py
ME=`basename "$0"`
#echo "$me"
HOMEPATH="D:"
# circles/finetune_test/"
EXPNUM="e059"
cp "$ME" "${HOMEPATH}/cherepashkin1/598test/plot_output/${EXPNUM}/${ME}"
#echo "$EXPNUM"
# python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 40  -bs 15 -num_input_images 3 -framelim 1000 -rescale 500 -cencrop 700 -framelim 1000 -criterion 'L1n' -localexp "" -lr 1e-3 -expnum "$EXPNUM" -hidden_dim 16 1500 -inputt "img"  -lb "pc" -zero_angle -gttype "single_f_n" -outputt "pc" -csvname "598csv9" -parallel "torch" -machine "workstation" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay 1e-4

python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 40  -bs 15 -num_input_images 3 -framelim 1000 -rescale 500 -cencrop 700 -framelim 1000 -criterion 'L1n' -localexp "" -lr 1e-4 -expnum "$EXPNUM" -hidden_dim 16 1500 -inputt "img"  -lb "pc" -zero_angle -gttype "single_f_n" -outputt "pc" -csvname "598csv9" -parallel "torch" -machine "workstation" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay 1e-4

python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 40  -bs 15 -num_input_images 3 -framelim 1000 -rescale 500 -cencrop 700 -framelim 1000 -criterion 'L1n' -localexp "" -lr 1e-6 -expnum "$EXPNUM" -hidden_dim 16 1500 -inputt "img"  -lb "pc" -zero_angle -gttype "single_f_n" -outputt "pc" -csvname "598csv9" -parallel "torch" -machine "workstation" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay 1e-4

EXPNUM="e060"
cp "$ME" "${HOMEPATH}/cherepashkin1/598test/plot_output/${EXPNUM}/${ME}"

python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 40  -bs 15 -num_input_images 3 -framelim 1000 -rescale 500 -cencrop 700 -framelim 1000 -criterion 'L1n' -localexp "" -lr 1e-3 -expnum "$EXPNUM" -hidden_dim 16 1500 -inputt "img"  -lb "pc" -zero_angle -gttype "single_f_n" -outputt "pc" -csvname "598csv9" -parallel "torch" -machine "workstation" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay 1e-5

python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 40  -bs 15 -num_input_images 3 -framelim 1000 -rescale 500 -cencrop 700 -framelim 1000 -criterion 'L1n' -localexp "" -lr 1e-3 -expnum "$EXPNUM" -hidden_dim 16 1500 -inputt "img"  -lb "pc" -zero_angle -gttype "single_f_n" -outputt "pc" -csvname "598csv9" -parallel "torch" -machine "workstation" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay 1e-6

python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 40  -bs 15 -num_input_images 3 -framelim 1000 -rescale 500 -cencrop 700 -framelim 1000 -criterion 'L1n' -localexp "" -lr 1e-3 -expnum "$EXPNUM" -hidden_dim 16 1500 -inputt "img"  -lb "pc" -zero_angle -gttype "single_f_n" -outputt "pc" -csvname "598csv9" -parallel "torch" -machine "workstation" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay 1e-7
