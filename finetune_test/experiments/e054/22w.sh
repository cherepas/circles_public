#!/bin/bash
python -u D:/circles/finetune_test/main.py -expdescr "truly normalized dataset" -bn 1000 -epoch 300 -bs 10 -lr 1e-3 -expnum e054 -cmscrop 550 -hidden_dim "hidden_dim = np.hstack((np.repeat(32, 1),1500))" -lb "pc" -criterion "L1" -inputt "img" -outputt "single_f_n" -num_input_images 1 -view_sep -csvname "598csv11" -no_rot_dirs -parallel "hvd" -pc_scale 100 -machine "workstation" -wandb "e054" -parallel "torch"
