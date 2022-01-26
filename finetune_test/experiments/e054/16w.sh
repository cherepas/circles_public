#WANDB_MODE=disabled
python -u D:/circles/finetune_test/main.py -expdescr "truly normalized dataset" -bn 1 -epoch 3 -bs 5 -lr 1e-2 -expnum e054 -cmscrop 550 -hidden_dim "hidden_dim = np.hstack((np.repeat(32, 1),1500))" -lb "pc" -criterion "L1" -inputt "img" -outputt "single_f_n" -num_input_images 1 -view_sep -csvname "598csv11" -no_rot_dirs -machine "workstation" -parallel "torch" -wandb "" 
