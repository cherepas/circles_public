WANDB_MODE=offline
WANDB_MODE=disabled
python -u D:/circles/finetune_test/main.py -expdescr "try trid-r2n2 for 3d reconstruction to voxels and then to points" -bn 10 -epoch 200 -bs 10 -lr 1e-2 -expnum e056 -cmscrop 550 -hidden_dim "hidden_dim = None" -lb "pc" -criterion "L1" -inputt "img" -outputt "single_f_n" -haf -num_input_images 3 -view_sep -csvname "598csv11" -no_rot_dirs -machine "workstation" -parallel "torch" -netname "tridr2n2" -use_existing_csv -use_sep_csv
