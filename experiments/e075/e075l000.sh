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
#export HOROVOD_MPI_THREADS_DISABLE=0
#source /p/home/jusers/cherepashkin1/jureca/cherepashkin1/virt_enves/venv1/activate.sh
# # make sure all GPUs on a node are visible
# export HOROVOD_MPI_THREADS_DISABLE=0
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python -u "../../main.py" -datapath "C:/cherepashkin1/phenoseed/" -realjobname `basename "$0"` -jobname `basename "$0"` -jobdir "$SCRIPT_DIR" -expnum "e075" -epoch 1000 -bs 50 -num_input_images 1 -framelim 6000 -criterion "L2" -rmdirname -lr 1e-5 -hidden_dim 9 -inputt "img" -outputt "orient" -lb "orient" -no_loadh5 -minmax_fn "" -parallel "torch" -machine "lenovo" -merging "batch" -aug_gt "" -updateFraction 0.5 -steplr 1000 1 -print_minibatch 1 -dfname "598frame" -num_workers 0 -minmax_fn "" -rescale 200