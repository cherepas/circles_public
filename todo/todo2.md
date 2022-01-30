# Improved appearance 
- print out not only opt, but also parameters from argparse from the sh file
  - Split line using regexp, that every parameter name starts with a minus sign. If it is a binary parameter (there is the next parameter with a minus sign), then just paste True or False in the column of values
- Draw a vertical line to split mini-batches one from each other
- Enable single seed loss output
- 

# Resultative improvements
- Organize special files with different if-else conditions for every step in training: 
  * prepare input data 
  * preparing output data
  * prepare Dataset output
- Compare center of masses for first 2500 point clouds, if it makes sense to subtract mean for them
# Automate manual postprocessing
- Render multiple md files as a single (useful for long tables)
- Synchronize plot output between devices to consolidate them together, but ignore large files, like the model, vector output, etc. 
- 
# Tips from meeting on 21.12.2021
- Regress only 6 elements of the upper-triangular matrix, which contains information about the pose. For every seed calculate second moments (MatLab script for normalization on canonical view). For the other views, firstly rotate the point cloud at exactly 10 degrees.
- At this configuration, the axis of rotation is vertical, so that one pose angle is fixed, and the other is incremented for rotation. Then calculate the second-moment matrix. After that ask network to regress the pose directly from images, as just 6 values in the output. Alternatively, one can try to regress two angles directly.
- The next step is to include information about the center of masses of the point cloud, this will be additionally 3 parameters.
- After this works with small latent space like 2 or 4, increase shape space.
- Basis function should be smoother
- Check what is 0,0,0 of every point cloud. Is it not the camera center?
- What is the vector towards the center of masses?
- Try to output 9 seeds (3x3)
- Show seed with most elongated part
- Plot legend, what is ground truth, and what is model output
# Plan for 14.01, 15.01, 17.01
- Prepare update report for 17.01
- Automatize several things to make update reports faster
  - Find csv2md on python
  - Save as a postprocessing file with functions to produce output automatically
  - 
- 
# Update report
## e066/000w - Ground truth from original point cloud

POse ground truth are just moments (X*X, X*Y, ... , Z*Z), calculated from the original point cloud without subtracting the mean, and without calculating eigenvalues 

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




# Questions
- Every seed rotates relatively to the holder tip. How to find holder tip position for every seed? 
- There is a difference if calculate moments from the original point cloud or point cloud with subtracted mean.![img_3.png](img_3.png) when I calculate mean, it already includes shape, because it depends on the shape, and center of masses is different from the tip position coordinate. The problem here is that I don't know the position of the tip in 3d or 2d. I only have camera center coordinates in 3d from the camera matrix. ![img_7.png](img_7.png)![img_8.png](img_8.png)![img_9.png](img_9.png)
- For every rotation angle should I regress
  - pose separately for every angle, process every rotation as a new entry (data-point)
  - regress pose and then rotate it on the known angle, and then calculate loss from the GT
- Should I regress 6 values directly from moments (X*X, X*Y, ..., Z*Z), or calculate the rotation matrix from finding eigenvalues? 



# Hanno tips on 17.01.2022
- Reconstruct not (XX, XY, ..), because it depends on size, but orientation matrix, gained from the decomposition of this matrix. 
  - Check How many points of freedom it has. It should be only 3, ![img_10.png](img_10.png), because 3 angle can describe every rotation. Another assumption, that the matrix is antisymmetrical. Is it? Or orthogonal?
  - Then try to add center of masses.
- Ask Andreas Fischbach to get the source code, how were point clouds generated. 
- If it find center and pose, it can do shape. Finally, 3D loss should include pose (and center of masses) loss
- Reread the article "What shape are dolphins". Classical fitting. 
  - Check how they track nose, flug, spine
- Check information about center of masses distribution, camera center distribution for the first 2500 seeds
- In 2. you may want to test, if using separate networks with shared weights -- one per image -- till the latent space and then joining them for pose regression may work better.



\![blabla](img_1.png)

# TODO from 18.01.2022 till 24.01.2022
- **Done** Use decomposed eigenvalues matrix as ground truth for regression of the pose from images
- **Done** Treat every rotation image as separate datapoint
- **in progress** Regress pose and center of masses from images
  - Shift point cloud to have it projected with the same projection matrix on every corresponding image.
- using separate networks with shared weights -- one per image -- till the latent space and then joining them for pose regression
- 
  - 
***
#Update report for 24.01.2022

## What was done
- Implement work with multiple images with color and batch channel
- Sort eigenvalues in ascending order to save them in the same way for different seeds
- Multiply orientation matrix on sign of determinant, to have the same sign of the matrix for different seeds
- Find calculation error, when trying to get **arcsin** from 1.0000000001.
- Orientation matrix consisting of eigenvectors is orthogonal.

![img_10.png](img_10.png)
Algorithm for Euler angles calculation from point cloud
- Subtracting mean
- Calculating coefficients of quadric form (X*X, X*Y, .., Z*Z)
- Calculating eigenvalues
- Sort eigenvalues and return corresponding matrix of eigenvectors
- Optionally for multiple views: divide matrix of eigenvalues by rotation matrix of corresponding angle
- Calculate Euler angles from matrix of eigenvectors to use them for GT
![img_16.png](img_16.png)

Rotation matrix from projection matrix consists of just from rotation around angle beta. 
![](../notebooks/figs/069.png)

Angles of orientation distributed differently, but there is big group that takes majority of orientations. 
![](../notebooks/figs/071.png)

['C:/circles/finetune_test/main.py', '-epoch', '1000', '-bs', '15', '-num_input_images', '1', '-framelim', '2500', '-rescale', '500', '-cencrop', '700', '-criterion', 'L1', '-localexp', '', '-lr', '1e-4', '-expnum', 'e066', '-hidden_dim', '3', '-inputt', 'img', '-outputt', 'eul', '-lb', 'eul', '-minmax_fn', 'min,max', '-zero_angle', '-gttype', 'single_f_n', '-csvname', '598csv9', '-parallel', 'torch', '-machine', 'lenovo', '-wandb', '', '-merging_order', 'color_channel', '-updateFraction', '0.25', '-steplr', '1000', '1', '-batch_output', '2', '-cmscrop', '0', '-weight_decay', '0', '-print_minibatch', '1']

parameter | value
--- | :---:
inputt|   img 
num_input_images  |   1 
outputt  |   eul 
lb  |   eul 
minmax_fn  |   min,max 
framelim  |   2500 
criterion  |   L1 
lr|   1e-4 
hidden_dim  |   3 
zero_angle |
updateFraction  |   0.25 

![img_1.png](C:/cherepashkin1/598test/plot_output/e066/078l/loss_out/Average_loss_Loss.png)

## e067/008w
parameter | value
--- | :---:
inputt|   img 
num_input_images  |   **3**
outputt  |   eul 
lb  |   eul 
minmax_fn  |   min,max 
framelim  |   **-** 
criterion  |   **L2** 
lr|   1e-4 
hidden_dim  |   3
merging order | **color_channel**
updateFraction  |   0.25
ellapsed time | 10 hours
![](../finetune_test/plot_output/e067/008w/loss_out/Average_loss_Loss.png)
![](../finetune_test/plot_output/e067/008w/loss_out/Average_loss_log10(Loss).png)

## e067/009w
parameter | value
--- | :---:
criterion | L1
lr | 1e-5
ellapsed time | 11 hours
![](../finetune_test/plot_output/e067/009w/loss_out/Average_loss_Loss.png)

## e067/010w
parameter | value
--- | :---:
num_input_images | 6
criterion | L1
ellapsed time | 1 day 7 hours

![](../finetune_test/plot_output/e067/010w/loss_out/Average_loss_Loss.png)

## e067/011w
parameter | value
--- | :---:
num_input_images | 12
criterion | L1
ellapsed time | in progress

![](../finetune_test/plot_output/e067/011w/loss_out/Average_loss_Loss.png)



# TODO from 24.01 to 31.01
- using separate networks with shared weights -- one per image -- till the latent space and then joining them for pose regression
- Regress pose and center of masses from images
  - Shift point cloud to have it projected with the same projection matrix on every corresponding image.


# Question to Hanno

- Does it make sense to regress ellipsoid from the images? Is it the same problem, as to regress 3 values for the pose, and 3 for the coordinate of the center
- Does it make sense to try regress pose from point cloud? It would be just a single matrix learn.

## Appendix

###TODO check that multiplication of rotation matrix C on orientation matrix will give the same result, as multiplication of C by point cloud. 

Sort eigenvalues to make them in ascending order. 
![img_12.png](img_12.png)
![img_13.png](img_13.png)

Multiply by determinant sign to make orientation uniform
![img_14.png](img_14.png)
For angle alpha there is weird plot 
![](../notebooks/figs/065.png)
![](../notebooks/figs/063.png)

 Measured statistic for rotation angles for 598: min, max, mean, std: 
 array([-3.14159265, 3.14159265, -0.31063423, 1.48487897])

![img_1.png](C:/cherepashkin1/598test/plot_output/e066/078l/loss_out/Average_loss_minibatch_train_Loss.png)

![img_1.png](C:/cherepashkin1/598test/plot_output/e066/078l/loss_out/Average_loss_minibatch_val_Loss.png)

Found that it is the same to first subtract mean and then rotate, or first rotate point cloud and then subtract mean. 

It is not the same, if calculate orientation matrix from rotated point cloud. 

-![img_11.png](img_11.png) is it ok, or antidiagonal elements should be not the same
- Determinant of orientation matrix is -1

# Hanno's tips on 24.01
- Permute GT vectors, as NN can accidentially flip axes. 
- If regress eul angles, take into account peridiocity. That -pi = 0 = pi for directions.
  - if regress 3x3 orientation matrix, multiplication on -1 should be the same.
  - Look how found minimum angles between two rays.
  - Pay attention, that eigenvectors is not orientation. We need orientation, not direction. So ray has two ends. 
- When evaluate model and regress 3x3 matrix, estimate closest orthogonal vector basis, because model output is not orthogonal
  - Calculate closest orthogonal during training can be computationally expensive
- Treat independent datapoints (in batch channel). Concatenate output of the network in last convolutional output of the network. Or in the first FC layer. Add additional FC layer of size 32 or 128, for example. 
- For ellipsoid fitting, read Direct least square fitting of ellipses https://ieeexplore.ieee.org/document/765658

#Update report for 31.01.2022
![](../notebooks/figs/089.png)

It takes 0.1 seconds for minibatch with size 5 and 3 views to calculate validation loss with augmentation. 

# Questions to Hanno:
- Can I just use min((loss(outputs, GT), loss(outputs, -GT))) in validation?
- 


