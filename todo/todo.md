# Improved appearance 
- print out not only opt, but also parameters from argparse from the sh file
  - Split line using regexp, that every parameter name starts with a minus sign. If it is binary parameter (there is the next parameter with a minus sign), then just paste True or False in the collumn of values
- Draw a vertical line to split minibatches one from each other
- Enable single seed loss output
- 

# Resultative improvements
- Organize special file with different if-else condition for every step in training: 
  * prepare input data 
  * preparing output data
  * prepare Dataset output
- Compare center of masses for first 2500 point clouds, if it make sense to subtract mean for them
# Automate manual postprocessing
- Render multiple md files as a single (usefull for long tables)
- Synchronize plot output between devices in order to consolidate them together, but ignore large files, like model, vector output, etc. 
- 
# Tips from meeting on 21.12.2021
- Try to output 9 seeds (3x3)
- Show seed with most elongated part
- Plot legend, what is ground truth, and what is model output
- Basis function should be smoother
- Regress only 6 elements of the upper-triangular matrix, which contains information about the pose. For every seed with calculate second moments (matlab script for normalization on canonical view). For the other views, firstly rotate the point cloud on exactly 10 degrees.
- At this configuration, axis of rotation is vertical, so that one pose angle is fixed, and the other is incremented for rotation. Then calculate the second-moment matrix. After that ask network to regress the pose directly from images, as just 6 values in the output. Alternatively, one can try to regress two angles directly. 
- The next step is to include information about the center of masses of the point cloud, this will be additionally 3 parameters. 
- After this works with small latent space like 2 or 4, increase shape space. 
- Check what is 0,0,0 of every point cloud. Is it not the camera center? 
- What is the vector towards center of masses? 
# Plan for 14.01, 15.01, 17.01
- Prepare update report for 17.01
- Automatize several things to make update report faster
  - Find csv2md on python
  - Save as a postprocessing file with functions to produce output automatically
  - 
- 
# Update report
## e066/000w - Ground truth from original point cloud
parameter | value
--- | :---:
bs|15
num_input_images|1
framelim|2000
rescale|500
cencrop|700
outputt|pose6
criterion|L1
lr|1e-4
hidden_dim|6
inputt|img
lb|pose6
zero_angle|
csvname|598csv9
parallel|torch
machine|workstation
![img](C:/cherepashkin1/598test/plot_output/e066/000w/learning_curve_log10(Loss).png)

## e066/001w
parameter | value
--- | :---:
bs|15
num_input_images|1
framelim|**2500**
rescale|500
cencrop|700
outputt|pose6
criterion|**L2**
lr|**5e-5**
steplr|**20, 0.2**
hidden_dim|6
inputt|img
lb|pose6
zero_angle|
csvname|598csv9
parallel|torch
machine|workstation
![img](C:/cherepashkin1/598test/plot_output/e066/001w/learning_curve_log10(Loss).png)

## e066/002w
parameter | value
--- | :---:
bs|15
num_input_images|1
framelim|**2500**
rescale|500
cencrop|700
outputt|pose6
criterion|**L2**
lr|**1e-3**
steplr|
hidden_dim|6
inputt|img
lb|pose6
zero_angle|
csvname|598csv9
parallel|torch
machine|workstation
![img](C:/cherepashkin1/598test/plot_output/e066/002w/learning_curve_log10(Loss).png)

## e066/005w - Ground truth is from point cloud with subtracted mean 
parameter | value
--- | :---:
bs|15
num_input_images|1
framelim|**2500**
rescale|500
cencrop|700
outputt|pose6
criterion|**L2**
lr|**1e-3**
hidden_dim|6
inputt|img
lb|pose6
zero_angle|
csvname|598csv9
parallel|torch
machine|workstation

![img](C:/cherepashkin1/598test/plot_output/e066/005w/loss_out/Average_loss_log10(Loss).png)

![img](C:/cherepashkin1/598test/plot_output/e066/005w/loss_out/Average_loss_minibatch_train_log10(Loss).png)

![img](C:/cherepashkin1/598test/plot_output/e066/005w/loss_out/Average_loss_minibatch_val_log10(Loss).png)

## e066/046l
parameter | value
--- | :---:
epoch|**10**
bs|**7**
num_input_images|1
framelim|**25**
rescale|500
cencrop|700
outputt|pose6
criterion|**L2**
lr|1e-5
hidden_dim|6
inputt|img
lb|pose6
zero_angle|
gttype|single_f_n
csvname|598csv9
parallel|torch
machine|lenovo
Here I have spikes between iterations. The first iteration here means that the neural network has seen the first minibatch. Then the first and the second and so on. 
![img_1.png](C:/cherepashkin1/598test/plot_output/e066/046l/Average_loss_minibatch_train_Loss.png)

![img_1.png](C:/cherepashkin1/598test/plot_output/e066/046l/Average_loss_minibatch_val_Loss.png)

POse ground truth are just moments (X*X, X*Y, ... , Z*Z), calculated from original point cloud without subtracting the mean, and without calculating eigen values 


# Questions
- For every rotation angle should I regress
  - pose separately for every angle, process every rotation as a new entry (data-point)
  - regress pose and then rotate it on the known angle, and then calculate loss from the GT
- There is difference, if calculate moments from original point cloud or from point cloud with subtracted mean.![img_3.png](img_3.png) when I calculate mean, it already includes shape, because it depends on the shape, and this is different from the tip position coordinate.The problem here that I don't know position of the tip in 3d or 2d. From camera matrix I have camera center coordinates in 3d. 
- Should I regress 6 values directly from moments (X*X, X*Y, ..., Z*Z), or calculate rotation matrix from finding eigen values? 
- 




\![blabla](img_1.png)