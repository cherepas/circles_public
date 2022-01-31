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
#HOROVOD_MPI_THREADS_DISABLE=0
#source /p/home/jusers/cherepashkin1/jureca/cherepashkin1/virt_enves/venv1/activate.sh
# # make sure all GPUs on a node are visible
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python -u "../../main.py" -datapath "D:/cherepashkin1/phenoseed/" -jobname `basename "$0"` -jobdir "$SCRIPT_DIR" -expnum "e072" -epoch 1 -bs 5 -num_input_images 3 -framelim 60 -criterion "L2" -localexp "" -lr 1e-4 -hidden_dim 9 -inputt "img" -outputt "orient" -lb "orient" -no_loadh5 -minmax_fn "" -parallel "torch" -machine "lenovo" -merging "batch" -aug_gt "orient" -updateFraction 0.25 -steplr 1000 1 -print_minibatch 10 -dfname "598frame"
