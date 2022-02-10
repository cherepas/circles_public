import argparse
parser = argparse.ArgumentParser()
def setarg(parser, argname, dfl):
    parser.add_argument('-'+argname, dest=argname,
                        action='store_true')
    parser.add_argument('-no_'+argname, dest=argname,
                        action='store_false')
    exec('parser.set_defaults('+argname+'=dfl)')
parser.add_argument('-bs', type=int, default=4)
parser.add_argument('-epoch', type=int, default=8)
parser.add_argument('-lr', type=float, default=5e-5)
# normalize output on the tmean matrix, to have min = 0 and max = 1
setarg(parser, 'minmax',False)
# normalize input point cloud to have every coordinate between 0 and 1
setarg(parser, 'minmax3dimage',False)
# normalize input point cloud, that it is in canonical view
setarg(parser, 'normalize',False)
# centerize input point cloud, to have it's center of masses in the origin
setarg(parser, 'center',False)
# linearly downsample input point cloud
parser.add_argument('-downsample', type=int, default=1)
# use f_n or f, that was gotten with normalization on canonical view before
# processing
setarg(parser, 'classicnorm',False)
# cut the number of maximum SH amplitude to regress
parser.add_argument('-ampl', type=int, default=441)
# centerize seed on the input image and crop to this width
parser.add_argument('-cmscrop', type=int, default=0)
parser.add_argument('-cencrop', type=int, default=700)
# rescale input image
parser.add_argument('-rescale', type=int, default=500)
setarg(parser, 'use_adasum',False)
parser.add_argument(
    '-gradient_predivide_factor', type=float, default=1.0,
    help='apply gradient predivide factor in optimizer (default: 1.0)')
# name of experiment directory
parser.add_argument('-expnum', type=str, default='111')
# hidden_dim - size of appendix FC layers
parser.add_argument(
    '-hidden_dim', nargs='+', type=int, default=[5000,2500,1000,441])
parser.add_argument(
    '-chidden_dim', nargs='+', type=int, default=[96, 128, 256, 256, 256])
parser.add_argument('-kernel_sizes', nargs='+', default=[7, 3, 3, 3, 3, 3])
# number of input images that will be loaded
parser.add_argument('-num_input_images', type=int, default=1)
# name of standard model
parser.add_argument('-model_name', type=str, default='')
parser.add_argument('-netname', nargs='+', default=['cnet'])
setarg(parser, 'use_pretrained',False)
parser.add_argument('-weight_decay', type=float, default=0)
# used to load images all in parallel, or merge them after output
# "separate" merging order means to get from Dataloader tensor like as for
# color channel, that [15, 3, 1000, 1800], but then reshape this tensor to
# the [45, 1, 1000, 1800] and work with it like with separate data points
parser.add_argument('-merging', type=str,
                    choices=['color', 'latent', 'batch'], default='batch')
# take input image of random angle, if not, then image will
# be taken relative to the horizontal pose
setarg(parser, 'rand_angle',False)
# number of experiment from phenoseeder
parser.add_argument('-specie', type=str, default='598')
# number of sampled directions to make subsampling after f_n
parser.add_argument('-num_sam_points', type=int, default=500)
# loss calculating between 'pc','f' or 'f_n'
parser.add_argument('-lb', type=str, default='f')
# short description what exactly this job is up for
parser.add_argument('-expdescr', type=str, default='')
# use csv file with pathes to all input files together with
# horizontal image index
setarg(parser, 'use_existing_csv',True)
setarg(parser, 'use_sep_csv',True)
# instead of input files noise is generating with random numbers
setarg(parser, 'noise_input',False)
# use convolutional part of the network or not
setarg(parser, 'haf',True)
# type of input data. can be 'img', 'f' or 'pc'
parser.add_argument('-inputt', type=str, default='img')
# normalize to make min = 0 and max = 1 for input f
setarg(parser, 'minmax_f',True)
# criterion to calculate loss
parser.add_argument('-criterion', type=str, default='L1')
# number of GPUs is used in the job
parser.add_argument('-ngpu', type=int, default=4)
# type of parallelization. 'hvd' means horovod, or 't'
parser.add_argument('-parallel', type=str, choices=['horovod', 'torch'],
                    default='hvd')
# in case loading standard model, it can be use as feature extracting
# (when freezeing all layers except the last one)
setarg(parser, 'feature_extract',False)
# if load only one image as input, this will be always image with index
# 000_rotation
# if load more than 1 image, then number of images will be spread evenly in
# the range (0,36)
# if false, images will be taking that first image in views will be with
# horizontal pose
setarg(parser, 'zero_angle',True)
# is used for testing computing time,
# where all needed files including data in one folder
parser.add_argument('-single_folder',
                    dest='single_folder', action='store_true')
parser.set_defaults(single_folder=False)
parser.add_argument('-noise_output', dest='noise_output',
                    action='store_true')
parser.set_defaults(noise_output=False)
# only log will be in the output
setarg(parser, 'save_output',True)
# type of data that is loaded for gt. for example, single_f_n
# means that only *f_n files will be used for GT in dataloader
# and maybe it will be singular loading of y_n
# it is used separate transform_f_n.py to not load more than is
# needed
# In case if gt is loaded not from dataloader, but from csv or from h5 file,
# there is option "single_file"
parser.add_argument('-gttype', type=str,
                    choices=['single_file'],
                    default='single_file')
# name of csv that will be used for loading GT
# it can be 598csv9 for original pose and 598csv11 for normalized pose
parser.add_argument('-csvname', type=str, default='598csv9')
# name of the csv which will be used for loading data
# choices are : 598frame for full or 598frame_dummy
parser.add_argument('-dfname', type=str, default='598frame')
# factor on which all output point cloud data will be normalized
parser.add_argument('-pscale', type=int, default=100)
# if view_sep = True, and more than one image is loaded,
# all input images will be treated as separate data elements
# new dataframe will be created
setarg(parser, 'view_sep',False)
# rotate directions together with angle from which
# current image were taken
setarg(parser, 'rot_dirs',False)
# for dataloader
parser.add_argument('-num_workers', type=int, default=0)
setarg(parser, 'pin_memory',False)
# manually calculate distance vector F out of point cloud output
setarg(parser, 'man_dist',False)
setarg(parser, 'use_cuda',True)
parser.add_argument('-machine', type=str,
                    choices=['jureca', 'workstation', 'lenovo', 'huawei'],
                    default='jureca')
setarg(parser, 'maintain',False)
setarg(parser, 'maintain_line',False)
parser.add_argument('-wandb', type=str, default="")
setarg(parser, 'measure_time',False)
setarg(parser, 'rotate_output',False)
parser.add_argument('-transappendix', type=str, default="_image")
# how often to save batch output intermediate in epoch
parser.add_argument('-batch_output', type=int, default=2)
# minmax fun for current ground truth preparation before training
parser.add_argument('-minmax_fn', type=str,
                    choices=['min,max','mean,std', ''], default='')
parser.add_argument('-updateFraction', type=float, default=3)
parser.add_argument('-standardize',  nargs='+', default=255)
# parser.add_argument('-standardize', default=(18.31589541, 39.63290785))
# if rmdirname is True, delete dirname content and use this directory again
# for saving output
setarg(parser, 'rmdirname', False)
parser.add_argument('-steplr',  nargs='+', type=float, default=(30,1))
parser.add_argument('-outputt', type=str,
                    choices=['points','pose6', 'eul', 'orient', 'cms'],
                    default='points')
parser.add_argument('-ufmodel', type=int, default=100000)
parser.add_argument('-framelim', type=int, default=int(1e20))
parser.add_argument('-conTrain', type=str, default='')
# how often to print loss in the log output
parser.add_argument('-print_minibatch', type=int, default=10)
# for orientation there are two right GT, because it is a ray. That is why
# augementation of ground truth is needed for evaluation
parser.add_argument('-aug_gt', nargs='+', type=str, default='')
parser.add_argument('-datapath', type=str,
                    default='C:/cherepashkin1/phenoseed')
# job name is used to create corresponding subdirectory
parser.add_argument('-jobname', type=str, default='')
# real job of the executed sh file. it is needed to copy sh file to the new
# directory
parser.add_argument('-realjobname', type=str, default='')
parser.add_argument('-jobdir', type=str, default='')
setarg(parser, 'loadh5', False)
opt = parser.parse_args()