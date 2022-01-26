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
HOMEPATH="C:"
# circles/finetune_test/"
EXPNUM="e066"
CONTRAIN="011l"
# mkdir "${HOMEPATH}/cherepashkin1/598test/plot_output/${EXPNUM}/"
#cp "$ME" "${HOMEPATH}/cherepashkin1/598test/plot_output/${EXPNUM}/${ME}"
#echo "$EXPNUM"
# 120 15 2000

#python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 120 -bs 15 -num_input_images 3 -framelim 500 -rescale 500 -cencrop 700 -criterion 'L1' -localexp "" -conTrain '000l' -lr 1e-3 -expnum "$EXPNUM" -hidden_dim 16 1500 -inputt "img" -lb "pc" -zero_angle -gttype "single_f_n" -outputt "pc" -csvname "598csv9" -parallel "torch" -machine "lenovo" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay
#python -u "${HOMEPATH}/cherepashkin1/598test/plot_output/e058/000w/main.py"

python -u "${HOMEPATH}/circles/finetune_test/main.py" -epoch 1 -bs 15 -num_input_images 1 -framelim 50 -rescale 500 -cencrop 700 -criterion 'L1' -localexp "" -lr 1e-3 -expnum "$EXPNUM" -hidden_dim 3 -inputt "img" -outputt "eul" -lb "eul" -minmax_fn "min,max" -zero_angle -gttype "single_f_n" -csvname "598csv9" -parallel "torch" -machine "lenovo" -wandb "" -merging_order 'color_channel' -updateFraction 1 -steplr 1000 1 -batch_output 2 -cmscrop 0 -weight_decay 0 -print_minibatch 1

#python -u "${HOMEPATH}/circles/finetune_test/main.py"
