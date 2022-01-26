#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
import argparse
import open3d
import torch
import horovod.torch as hvd
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
from skimage import io, transform
from skimage.measure import regionprops
from skimage import filters
from torch.utils.data import Dataset
import pandas as pd
from scipy.special import sph_harm
import sys
from numpy import linalg as LA
import datetime
from standard_models import *
import torch.multiprocessing as mp
import shutil
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('-bs', type=int, default=4)
parser.add_argument('-epoch', type=int, default=8)
parser.add_argument('-bn', type=int, default=8)
parser.add_argument('-lr', type=float, default=5e-5)
# normalize output on the tmean matrix, to have min = 0 and max = 1
parser.add_argument('-minmax', dest='minmax', action='store_true')
parser.add_argument('-no_minmax', dest='minmax', action='store_false')
parser.set_defaults(minmax=False)
# normalize input point cloud to have every coordinate between 0 and 1
parser.add_argument(
    '-minmax3dimage', dest='minmax3dimage', action='store_true')
parser.add_argument(
    '-no_minmax3dimage', dest='minmax3dimage', action='store_false')
parser.set_defaults(minmax3dimage=False)

#normalize input point cloud, that it is in canonical view
parser.add_argument('-normalize', dest='normalize', action='store_true')
parser.add_argument('-no_normalize', dest='normalize', action='store_false')
parser.set_defaults(normalize=False)

#centerize input point cloud, to have it's center of masses in the origin
parser.add_argument('-center', dest='center', action='store_true')
parser.add_argument('-no_center', dest='center', action='store_false')
parser.set_defaults(center=False)

#linearly downsample input point cloud
parser.add_argument('-downsample', type=int, default=1)

#use f_n or f, that was gotten with normalization on canonical view before processing
parser.add_argument('-classicnorm', dest='classicnorm', action='store_true')
parser.add_argument(
    '-no_classicnorm', dest='classicnorm', action='store_false')
parser.set_defaults(classicnorm=False)

#cut the number of maximum SH amplitude to regress
parser.add_argument('-ampl', type=int, default=441)

#centerize seed on the input image and crop to this width
parser.add_argument('-cmscrop', type=int, default=550)
#rescale input image
parser.add_argument('-rescale', type=int, default=550)

parser.add_argument('-use_adasum', type=bool, default=False)
parser.add_argument(
    '-gradient_predivide_factor', type=float, default=1.0,
    help='apply gradient predivide factor in optimizer (default: 1.0)')

#name of experiment directory
parser.add_argument('-expnum', type=str, default='111')
# parser.add_argument('-hidden_dim',type=str,
#   default = 'hidden_dim = np.hstack((np.repeat(128, 3),441))')

# hidden_dim - size of appendix FC layers
parser.add_argument(
    '-hidden_dim', type=str,
    default='hidden_dim=np.array([5000,2500,1000,441])')
parser.add_argument(
    '-chidden_dim', type=str,
    default='chidden_dim = np.hstack((96,128,np.repeat(256, 3)))')
parser.add_argument(
    '-kernel_sizes', type=str,
    default='kernel_sizes = np.hstack((7,np.repeat(3, 5)))')
#number of input images that will be loaded
parser.add_argument('-num_input_images', type=int, default=1)
#name of standard model
parser.add_argument('-model_name', type=str, default='')
parser.add_argument(
    '-use_pretrained', dest='use_pretrained', action='store_true')
parser.add_argument(
    '-no_use_pretrained', dest='use_pretrained', action='store_false')
parser.set_defaults(use_pretrained=False)
parser.add_argument('-weight_decay', type=float, default=0)

#used to load images all in parallel, or merge them after output
parser.add_argument('-merging_order', type=str, default='')
#take input image of random angle, if not, then image will 
#be taken relative to the horizontal pose
parser.add_argument('-rand_angle', dest='rand_angle', action='store_true')
parser.add_argument('-no_rand_angle', dest='rand_angle', action='store_false')
parser.set_defaults(rand_angle=False)
#number of experiment from phenoseeder
parser.add_argument('-specie', type=str, default='598')
#number of sampled directions to make subsampling after f_n
parser.add_argument('-num_sam_points', type=int, default=500)
#loss calculating between 'pc','f' or 'f_n'
parser.add_argument('-lb', type=str, default='f')
#short description what exactly this job is up for
parser.add_argument('-expdescr', type=str, default='')
#use csv file with pathes to all input files together with
# horizontal image index
parser.add_argument(
    '-use_existing_csv', dest='use_existing_csv', action='store_true')
parser.add_argument(
    '-no_use_existing_csv', dest='use_existing_csv', action='store_false')
parser.set_defaults(use_existing_csv=True)
#instead of input files noise is generating with random numbers
parser.add_argument('-noise_input', dest='noise_input', action='store_true')
parser.set_defaults(noise_input=False)
#use convolutional part of the network or not
parser.add_argument('-haf', dest='haf', action='store_true')
parser.add_argument('-no_haf', dest='haf', action='store_false')
parser.set_defaults(haf=True)
#type of input data. can be 'img', 'f' or 'pc'
parser.add_argument('-inputt', type=str, default='img')
#normalize to make min = 0 and max = 1 for input f
parser.add_argument('-minmax_f', dest='minmax_f', action='store_true')
parser.add_argument('-no_minmax_f', dest='minmax_f', action='store_false')
parser.set_defaults(minmax_f=True)
#criterion to calculate loss
parser.add_argument('-criterion', type=str, default='L1')
#number of GPUs is used in the job
parser.add_argument('-ngpu', type=int, default=4)
#type of parallelization. 'hvd' means horovod, or 'torch'
parser.add_argument('-parallel', type=str, default='hvd')
#in case loading standard model, it can be use as feature extracting
#(when freezeing all layers except the last one)
parser.add_argument(
    '-feature_extract', dest='feature_extract', action='store_true')
parser.add_argument(
    '-no_feature_extract', dest='feature_extract', action='store_false')
parser.set_defaults(feature_extract=False)
#if load only one image as input, this will be always image with index 000_rotation
parser.add_argument('-zero_angle', dest='zero_angle', action='store_true')
parser.add_argument('-no_zero_angle', dest='zero_angle', action='store_false')
parser.set_defaults(zero_angle=False)
#is used for testing computing time, 
#where all needed files including data in one folder
parser.add_argument('-single_folder',
                    dest='single_folder', action='store_true')
parser.set_defaults(single_folder=False)

parser.add_argument('-noise_output', dest='noise_output',
                    action='store_true')
parser.set_defaults(noise_output=False)
#only log will be in the output
parser.add_argument('-no_save_output', dest='save_output',
                    action='store_false')
parser.set_defaults(save_output=True)
#type of data that is loaded for gt. for example, single_f_n
#means that only *f_n files will be used for GT in dataloader
#and maybe it will be singular loading of y_n
#it is used separate transform_f_n.py to not load more than is
#needed
parser.add_argument('-outputt', type=str, default='single_f_n')
#name of csv that will be used for loading GT
parser.add_argument('-csvname', type=str, default='598csv9')
#factor on which all output point cloud data will be normalized
parser.add_argument('-pc_scale', type=int, default=100)

#if view_sep = True, and more than one image is loaded, 
# all input images will be treated as separate data elemets
parser.add_argument('-view_sep', dest='view_sep',
                    action='store_true')
parser.add_argument('-no_view_sep', dest='view_sep',
                    action='store_false')
parser.set_defaults(view_sep=True)

#rotate directions together with angle from which 
#current image were taken
parser.add_argument('-rot_dirs', dest='rot_dirs',
                    action='store_true')
parser.add_argument('-no_rot_dirs', dest='rot_dirs',
                    action='store_false')
parser.set_defaults(view_sep=False)

#for dataloader
parser.add_argument('-num_workers', type=int, default=0)

parser.add_argument('-pin_memory', dest='pin_memory',
                    action='store_true')
parser.add_argument('-no_pin_memory', dest='pin_memory',
                    action='store_false')
parser.set_defaults(pin_memory=False)

#manually calculate distance vector F out of point cloud output
parser.add_argument('-man_dist', dest='man_dist',
                    action='store_true')
parser.add_argument('-no_man_dist', dest='man_dist',
                    action='store_false')
parser.set_defaults(man_dist=False)
opt = parser.parse_args()

nim = opt.num_input_images
enim = nim if not opt.view_sep else 1
nsp = opt.num_sam_points
tstart = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vox2mm = 0.05
if opt.parallel == 'hvd':
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.set_num_threads(1)
    rank = hvd.rank()
else:
    rank = 0
if rank == 0:
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
classicnorm = '_prenormalized' if opt.classicnorm else ''

if '619' in opt.specie:
    original_h, original_w = 2048, 2448
else:
    original_h, original_w = 1000, 1800
homepath = '/p/home/jusers/cherepashkin1/jureca/'
path1 = os.path.join(homepath, 'circles/finetune_test')
os.chdir(path1)

if opt.single_folder:
    from cnet import *
else:
#     exec(open(os.path.join(['/p/home/jusers/cherepashkin1/',
#                             'jureca/cherepashkin1/experiments'],
#                             opt.expnum+'cnet.py')).read())
    exec('from experiments.'+opt.expnum+'.'+'cnet import *')


def my_loss(output, target):
    myloss = torch.mean(torch.multiply(weightv, (output - target))**2)
    return myloss

if not opt.single_folder:
    mainpath = os.path.join(homepath, 'cherepashkin1/phenoseed_csv')
else:
#     print('__file__', __file__)
#     mainpath = os.path.join(os.path.abspath(__file__).replace('smain.py',''),'phenoseed_csv')
    mainpath = os.path.join(homepath, 'seed2sh')
    

dir1 = homepath +\
    'cherepashkin1/598test/plot_output/' + opt.expnum
if opt.save_output and rank == 0 and not os.path.exists(dir1):
    os.mkdir(dir1)


def newfold(dir1):
    i = 0
    while True:
        n = str(i)
        dirname = dir1+'/'+n.zfill(3)+'/'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            break
        else:
            i += 1
            continue
    return(dirname)
parameters_list = (
    'expdescr: ' + opt.expdescr + '\n' +
    'epoch='+str(opt.epoch) + '\n' +
    'bn='+str(opt.bn) + '\n' +
    'bs='+str(opt.bs) + '\n' +
    'lr='+str(opt.lr) + '\n' +
    'minmax=' + str(opt.minmax) + '\n' +
    'minmax3dimage=' + str(opt.minmax3dimage) + '\n' +
    'downsample=' + str(opt.downsample) + '\n' +
    'normalize=' + str(opt.normalize) + '\n' +
    'center=' + str(opt.center) + '\n' +
    'classicnorm=' + str(opt.classicnorm) + '\n' +
    'ampl=' + str(opt.ampl) + '\n' +
    'cmscrop=' + str(opt.cmscrop) + '\n' +
    'rescale=' + str(opt.rescale) + '\n' +
    'use_adasum=' + str(opt.use_adasum) + '\n' +
    'gradient_predivide_factor=' +
    str(opt.gradient_predivide_factor) + '\n' +
    'expnum=' + str(opt.expnum) + '\n' +
    'hidden_dim=' + opt.hidden_dim + '\n' +
    'chidden_dim=' + opt.chidden_dim + '\n' +
    'kernel_sizes=' + opt.kernel_sizes + '\n' +
    'inputt == pc, img_input, F input? =' +
    str(opt.inputt == 'pc') + str(opt.inputt == 'img') +
    str(opt.inputt == 'f') + '\n' +
    'model_name=' + opt.model_name + '\n' +
    'feature_extract=' + str(opt.feature_extract) + '\n' +
    'Was it used pretrained model ? =' +
    str(opt.use_pretrained) + '\n' +
    'weight_decay=' + str(opt.weight_decay) + '\n' +
    'num_input_images=' + str(nim) + '\n' +
    'merging_order=' + str(opt.merging_order) + '\n' +
    'specie=' + opt.specie + '\n' +
    'number of sampled points=' + str(nsp) + '\n' +
    'loss is between model output and gt of =' +
    opt.lb + '\n' +
    'rand_angle=' + str(opt.rand_angle) + '\n' +
    'zero_angle=' + str(opt.zero_angle) + '\n' +
    'ngpu=' + str(opt.ngpu) + '\n' +
    'outputt=' + opt.outputt + '\n' +
    'csvname=' + opt.csvname + '\n' +
    'pc_scale=' + str(opt.pc_scale) + '\n' +
    'view_sep=' + str(opt.view_sep) + '\n' +
    'rot_dirs=' + str(opt.rot_dirs) + '\n' +
    'was there just noise as input? =' + str(opt.noise_input))

lb = opt.lb
epoch0 = -1
def train_model(model, dataloaders, samplers, criterion, optimizer,
                tmean, mainpath, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    lossar = np.zeros([2, num_epochs])
    y_n = np.genfromtxt(os.path.join(
        mainpath, 'Y_N_' +
        str(nsp) +
        classicnorm+'.csv').replace('\\', '/'),
        delimiter=',')
    bX = np.genfromtxt(os.path.join(
        mainpath, 'bX_' +
        str(nsp) +
        classicnorm+'.csv').replace('\\', '/'),
        delimiter=',')

    prmat = np.genfromtxt(os.path.join(
        mainpath, 'prmat.csv').replace('\\', '/'),
        delimiter=',')
    C = np.zeros([36,3,3])
    E = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0]]
    for i in range(36):
        C[i,:,:] =\
        np.matmul(np.matmul(E,prmat[4*i:4*(i+1),:]),
            np.linalg.pinv(np.matmul(E,prmat[0:4,:])))
    bX = torch.Tensor(bX).cuda()
    bX = torch.transpose(bX, 0, 1)
    C = torch.Tensor(C).cuda()
    y_n = torch.Tensor(y_n).cuda()
    tmeant = torch.Tensor(tmean).cuda()
    cnt = 0
    for epoch in range(num_epochs):
        if rank == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            rloss = 0.0
            if opt.parallel == 'hvd':
                train_sampler = samplers[phase]
                train_sampler.set_epoch(epoch)
            # Iterate over data.
            loss_cnt = 0
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                ts = time.time()
                if i_batch == bn:
                    break
                f_n = sample_batched['f_n']
                angles_list = sample_batched['angles']
#                 print(angles_list)
                pathes = sample_batched['path']
                if lb == 'f' and not opt.outputt == 'single_f_n':
                    far = sample_batched['far']
                if opt.inputt == 'img' or opt.inputt == 'pc':
                    inputs = sample_batched['image']
                elif opt.inputt == 'f':
                    inputs = far
                # zero the parameter gradients
                optimizer.zero_grad()
                tf = time.time()
#                 print('loading time = ,',tf-ts)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if opt.inputt == 'pc':
                        x = inputs
#                         y = torch.zeros([x.shape[0], x.shape[1]])
                        y = torch.norm(x, dim=2)
#                         for i in range(x.shape[0]):
#                             y[i, :] = torch.sqrt(
#                                 x[i, :, 0]**2 +
#                                 x[i, :, 1]**2 + x[i, :, 2]**2)
                        y = torch.unsqueeze(y, 2)
                        y = y.cuda()
                        outputs = model(y)
                    elif (opt.inputt == 'img' or\
                            opt.inputt == 'f') and\
                            not lb == 'pc+f':
                        ts = time.time()
                        outputs = model(inputs)
                        tf = time.time()
#                         print('model time = ',tf-ts)
                    elif (opt.inputt == 'img' or\
                            opt.inputt == 'f') and\
                            lb == 'pc+f':
                        outputs, outputs1 = model(inputs)
                    if lb == 'f' and\
                            opt.minmax:
                        ampl = opt.ampl
                        outputs = torch.multiply(
                            outputs, tmeant[3, :ampl] -
                            tmeant[2, :ampl]) + tmeant[2, :ampl]
                    if lb == 'f':
                        F = torch.zeros(
                            inputs.shape[0], nsp).cuda()
                    if (lb == 'pc' or\
                        lb == 'f' or lb == 'pc+f') and\
                        enim == 1 and\
                            not(opt.zero_angle or opt.rand_angle) and opt.rot_dirs:
                        y_n2 = torch.zeros(inputs.shape[0],
                                           nsp,
                                           opt.ampl).cuda()
                        for i, angle in enumerate(angles_list):
                            y_n2[i] = \
                                y_n[int(angle/10)*nsp:
                                    (int(angle/10) +
                                     1)*nsp, :]
                    elif (lb == 'f' or lb == 'pc' or lb == 'pc+f') and\
                        enim == 1 and\
                            (opt.zero_angle or not opt.rot_dirs):
                        y_n2 = \
                            torch.unsqueeze(y_n[:nsp,
                                                :], axis=0)
                        y_n2 = \
                            y_n2.repeat(inputs.shape[0], 1, 1)
                    if (lb == 'pc' or lb == 'f' or lb == 'pc+f'):
                        dirs = torch.zeros(inputs.shape[0],
                                           nsp,
                                           2).cuda()
                    if (lb == 'pc' or\
                            lb == 'f' or lb == 'pc+f') and opt.rot_dirs:
                        for i, angle in enumerate(angles_list):
                            dirs[i] = \
                                bX[:, int(angle/10)*2:
                                    (int(angle/10) +
                                     1)*2]
                    elif (lb == 'pc' or\
                            lb == 'f' or lb == 'pc+f') and not opt.rot_dirs:
                        dirs = bX[:, :2].repeat(inputs.shape[0], 1, 1)
#                         for i, angle in enumerate(angles_list):
#                             dirs[i] = \
#                                 bX[:, :2]                        
                    if not enim == 1:
                        print('Допили скрипт для работы\
                              с несколькими изображениями')
                        sys.exit()
                    if lb == 'f':
                        for i in range(inputs.shape[0]):
                            F[i] = torch.matmul(y_n2[i], outputs[i])
                    elif lb == 'pc+f':
                        p1 = f2p(y_n2, outputs, dirs, nsp) 
                    if lb == 'f_n':
                        loss = criterion(outputs, f_n)
                    if lb == 'pc' or lb == 'f':
                        #making F GT from f_n gt
                        far = torch.zeros(inputs.shape[0],nsp).cuda()
                        for i in range(inputs.shape[0]):
                            far[i] = torch.matmul(y_n2[i], f_n[i])                        
                    if lb == 'pc' or lb == 'pc+f':
                        # making pc GT
                        p = f2p(y_n2, f_n, dirs, nsp)
                        for i in range(inputs.shape[0]):
                            p[i,:,:] = torch.matmul(
                                torch.transpose(torch.squeeze(
                                    C[int(angles_list[i]/10),:,:]),0,1),p[i,:,:])
#                         print(p.shape)
                    if lb == 'pc' and opt.criterion == 'L2':
                        loss = torch.sqrt(
                            torch.mean((p.view(inputs.shape[0], -1)\
                                       -outputs*opt.pc_scale) ** 2)) /\
                            1.1546765389234168
                    elif lb == 'pc' and opt.criterion == 'L1':
                        loss = torch.mean(torch.abs(p.view(inputs.shape[0], -1)\
                                       -outputs*opt.pc_scale))
                    elif lb == 'f' and\
                            opt.criterion == 'L2':
                        loss = torch.sqrt(
                            torch.mean((far-F) ** 2)
                            ) /\
                            1.1546765389234168
                    elif lb == 'f' and\
                            opt.criterion == 'L1':
                        loss = torch.mean(
                            torch.abs(far-F))
                    elif lb == 'pc+f' and\
                            opt.criterion == 'L1':
#                         loss0 = torch.mean(
#                             torch.abs(far-F))
                        loss0 = torch.mean(torch.abs(p.view(inputs.shape[0], -1)\
                                       -p1))                        
                        loss1 = torch.mean(torch.abs(p.view(inputs.shape[0], -1)\
                                       -outputs1*opt.pc_scale))
                        print('loss0=',loss0)
                        print('loss1=',loss1)
                        loss = loss0+loss1
#                     _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                loss_cnt += 1
                rloss += loss.item()
            if bn*opt.bs < len(dataloaders[phase].dataset):
                ebn = bn*opt.bs
            else:
                ebn = len(dataloaders[phase].dataset)
            if ebn == 0:
                ebn = 1
            if phase == 'train' and lb == 'pc':
                lossar[0][epoch] = rloss/loss_cnt*vox2mm
            elif phase == 'val' and lb == 'pc':
                lossar[1][epoch] = rloss/loss_cnt*vox2mm
            if phase == 'train' and not lb == 'pc':
                lossar[0][epoch] = rloss/loss_cnt
            elif phase == 'val' and not lb == 'pc':
                lossar[1][epoch] = rloss/loss_cnt
#                 if lb == 'pc':
#                 lossar[:][epoch] *= vox2mm
            log1 = (rank == 0 and
                    epoch > epoch0 and
                    opt.save_output and
                    (lb == 'f' or
                     lb == 'pc' or lb == 'pc+f'))
            if epoch == epoch0+1 and opt.save_output and rank == 0:
                dirname = None
                while dirname is None:
                    try:
                        dirname = newfold(dir1)
                    except:
                        time.sleep(1)
                        pass
#                 else:
#                     dirname = os.path.join(dir1, 'misc')
                original_stdout = sys.stdout
                with open(dirname+"opt.txt", 'a') as f:
                    sys.stdout = f  # Change the standard output to the file we created.
                    print(opt)
                with open(dirname+"sys_argv.txt", 'a') as f:
                    sys.stdout = f  # Change the standard output to the file we created.
                    print(sys.argv)
                sys.stdout = original_stdout

                print('opt:\n',opt)
                print('sys.argv:\n',sys.argv)

                with open(dirname+"job-parameters.txt", "w") as f:
                    f.write(parameters_list)                 
            if log1 and opt.minmax:
                ampl = opt.ampl
                o = np.multiply(
                    outputs.detach().cpu().numpy(),
                    tmean[3, :ampl] -
                    tmean[2, :ampl]) + tmean[2, :ampl]
                gt = np.multiply(
                    f_n.detach().cpu().numpy(),
                    tmean[3, :ampl] -
                    tmean[2, :ampl]) + tmean[2, :ampl]
            elif log1 and not opt.minmax:
                o = outputs.detach().cpu().numpy()
            if rank == 0 and opt.save_output and\
                    lb == 'pc+f' and not opt.minmax and epoch > epoch0:
                o = p1.detach().cpu().numpy()*vox2mm
                oo = o[0]
                o = np.reshape(o,(o.shape[0],-1))
            if rank == 0 and opt.save_output and\
                    lb == 'pc' and not opt.minmax and epoch > epoch0:
                o = o*opt.pc_scale
            if log1 and not opt.minmax and\
                    (lb == 'f') and epoch == epoch0+1:
                gt = f_n.detach().cpu().numpy()
            if rank == 0 and opt.save_output and\
                    (lb == 'pc' or lb == 'pc+f') and not opt.minmax and\
                    epoch == epoch0+1:
                gt = p.detach().cpu().numpy()*vox2mm
                gt0 = gt[0]
                gt = np.reshape(gt,(gt.shape[0],-1))
#             if rank == 0 and opt.save_output and\
#                     (lb == 'pc' and epoch == epoch0+1:
#                 gt0 = gt[0].reshape((3, nsp))
            if rank == 0 and opt.save_output and\
                    lb == 'pc' and epoch > epoch0:
                o *= vox2mm
            if log1 and phase == 'val':
                lossars = np.array([np.trim_zeros(lossar[0,:]),
                    np.trim_zeros(lossar[1,:])])
#                 print(lossars)
                lossfig(dirname, lossars, 'Loss')
                logloss = np.ma.log10(lossars)
                lossfig(dirname, logloss.filled(0), 'log10(Loss)')
            if (rank == 0 and
                    epoch == epoch0+1 and
                    opt.save_output and
                    (lb == 'f' or
                     lb == 'pc' or lb == 'pc+f')):
                with open(dirname+"pathes_"+phase+".txt", "w") as f:
                    for element in pathes:
                        f.write(element + "\n")
            if log1 and (lb == 'pc' or lb == 'pc+f') and epoch == epoch0+1:
                np.savetxt(dirname + 'gt',
                           np.reshape(gt,(inputs.shape[0],-1)), delimiter=',')   
            if rank == 0 and epoch == epoch0+1 and opt.save_output:
                shutil.copy(os.path.join(homepath, 'circles/finetune_test/experiments',
                                         opt.expnum, 'cnet.py'), dirname + 'cnet.py')
            if log1 and lb == 'f' and epoch == epoch0+1:
                p = f2p(y_n2, f_n, dirs, nsp)
                gt0 = p[0].detach().cpu().numpy()*vox2mm
            if log1 and lb == 'f': 
                p1 = f2p(y_n2, outputs, dirs, nsp)
#                 p1 = torch.zeros(\
#                     y_n2.shape[0],3,nsp).cuda()
                
#                 print('p1.shape',p1.shape)
#                 for i in range(y_n2.shape[0]):
# #                     far[i] = torch.matmul(y_n2[i], f_n[i])
#                     print(outputs.shape, dirs.shape)
#                     p1[i,0,:]=outputs[i]*torch.cos(dirs[i,:,0])*\
#                         torch.sin(dirs[i,:,1])
#                     p1[i,1,:]=outputs[i]*torch.sin(dirs[i,:,0])*\
#                         torch.sin(dirs[i,:,1])
#                     p1[i,2,:]=outputs[i]*torch.cos(dirs[i,:,1])
                oo = p1[0].detach().cpu().numpy()*vox2mm
            if (rank == 0 and
                    epoch > epoch0 and
                    opt.save_output and
                    (lb == 'f' or
                     lb == 'pc' or lb == 'pc+f')):
                np.savetxt(dirname + 'o_'+phase+'_' +\
                           str(cnt).zfill(3), o, delimiter=',')
                np.save(dirname + 'lossar.npy', lossar)

                torch.save(model.state_dict(), dirname+"model")
            if log1 and lb == 'f':
#cnt = shplot(o[0], dirname, cnt)
                np.savetxt(dirname + 'far',
                   far.detach().cpu().numpy(), delimiter=',')
            if rank == 0 and (lb == 'pc') and\
                opt.save_output and epoch > epoch0:
                oo = o[0]
                oo = oo.reshape((3, nsp))
            if rank == 0 and (lb == 'pc' or lb == 'f' or lb =='pc+f') and\
                opt.save_output and epoch > epoch0:
                #                 print(gt.shape)
#                 gt0 = gt[0]
#                 gt0 = gt.reshape((3, nsp))
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(oo[0, :], oo[1, :], oo[2, :], c='r')
                ax.scatter(gt0[0, :], gt0[1, :], gt0[2, :], c='g')
                ax_lim = 60*vox2mm
                ax.set_xlim(-ax_lim, ax_lim)
                ax.set_ylim(-ax_lim, ax_lim)
                ax.set_zlim(-ax_lim, ax_lim)
                plt.savefig(dirname + 'pc_'+phase+'_' +\
                            str(cnt).zfill(3) + '.png')
            if rank == 0 and opt.save_output and lb == 'pc' and epoch > epoch0:
                cip = []
                for root, directories, filenames in os.walk(dirname): 
                    for filename in filenames:
                        if 'pc_'+phase in filename and not 'checkpoint' in filename:
                            cip.append(os.path.join(root,filename))
                cip.sort()
                images = []
                for filename in cip:
                    images.append(imageio.imread(filename))
                imageio.mimsave(os.path.join(dirname, phase+'_movie.gif'), images, duration=0.5)
            losst = 0 if phase == 'train' else 1
            if rank == 0:
                print('{} Loss: {:.6f}'.format(phase, lossar[losst][epoch]))
                print()
        cnt += 1
    time_elapsed = time.time() - since
    if rank == 0:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    return model, lossar, time_elapsed, dirname
def f2p(y_n, f_n, dirs, nsp):
    p = torch.zeros(\
        y_n.shape[0],3,nsp).cuda()
    for i in range(y_n.shape[0]):
        far = torch.matmul(y_n[i], f_n[i])
        p[i,0,:]=far*torch.cos(dirs[i,:,0])*\
            torch.sin(dirs[i,:,1])
        p[i,1,:]=far*torch.sin(dirs[i,:,0])*\
            torch.sin(dirs[i,:,1])
        p[i,2,:]=far*torch.cos(dirs[i,:,1])
    return p

def lossfig(dirname, lossar, ylabel):
    plt.rcParams["figure.figsize"] = (9, 5)
    fig = plt.figure()
    axes = plt.gca()
    labels_text = ['train', 'val']
    for i in range(2):
        plt.plot(
            np.arange(lossar.shape[1]), lossar[i, :],
            label=labels_text[i], linewidth=3)
    plt.grid()
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='best', borderaxespad=0., fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.autoscale(enable=True, axis='both', tight=None)
    axes = plt.gca()
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=4)
    plt.savefig(
        dirname+'learning_curve_'+ylabel+'.png', bbox_inches='tight')
    plt.savefig(
        dirname+'learning_curve_'+ylabel+'.pdf', bbox_inches='tight')


class Seed3D_Dataset(Dataset):
    """seed point cloud dataset."""

    def __init__(self, mainpath, lframe, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = lframe
        self.root_dir = mainpath
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = 0
        if opt.inputt == 'pc':
            img_name = os.path.join(self.root_dir, self.root_dir.replace(
                'phenoseed_csv', 'phenoseed'),
                self.df.iloc[idx, 0].replace('csv', '')+
                '_Surface.ply').replace('\\', '/')
            pcd = open3d.io.read_point_cloud(img_name)
            img = np.asarray(pcd.points)
            img = np.concatenate(
                (img, np.zeros([58014 - img.shape[0], 3])), axis=0)
        elif opt.inputt == 'img':
            torch.manual_seed(idx)
        if opt.inputt == 'img' and opt.noise_input:
            img = torch.rand(enim, original_h, original_w)
            angles_list = np.array([0])
#         print(inn)
        if inn and not(opt.rand_angle or opt.view_sep or opt.zero_angle):
            angles_list = np.zeros(alf.shape)
            for i, angle in enumerate(alf):
                newa = angle - 10*int(self.df.iloc[idx, 1])
                angle = newa if newa >= 0 else newa+360
                angles_list[i] = int(angle)
        elif inn and not(opt.rand_angle or opt.view_sep) and opt.zero_angle:
            angles_list = alf
        elif inn and not opt.rand_angle and opt.view_sep:
            angles_list = np.array([10*self.df.iloc[idx,1]])
#         for i in range(len(angles_list)):
#             angles_list[i] = int(angles_list[i])
        angles_list = angles_list.astype(int)
#         print(angles_list)
        if inn:
            img_name = []
            img = np.zeros([enim, original_h, original_w])
            for i in range(enim):
                img_name = os.path.join(
                    self.root_dir.replace('phenoseed_csv', 'phenoseed'),
                    self.df.iloc[idx, 0].replace(opt.csvname, opt.specie),
                    'rotation_' + str(angles_list[i]).zfill(3) +
                    '.tif').replace('\\', '/')
#                 img[i] = np.expand_dims(np.asarray(io.imread(img_name)),
#                                                         axis=2)
                curim = np.asarray(io.imread(img_name))
                h1, w1 = curim.shape
                if (h1, w1) == (original_h, original_w):
                    img[i] = curim
                else:
                    h2, w2 = original_h, original_w
                    th, tw = int((h2-h1) / 2), int((w2-w1) / 2)
                    img[i] = np.pad(curim, ((th, th), (tw, tw)))
            img = 1 - img/255
        if not opt.noise_output and not opt.outputt == 'single_f_n':
            fara = np.genfromtxt(os.path.join(self.root_dir, (
                self.df.iloc[idx, 0] +
                '_Far_' + str(nsp) + classicnorm +
                '.csv')).replace('\\', '/'), delimiter=',')
            f_na = np.genfromtxt(os.path.join(self.root_dir, (
                self.df.iloc[idx, 0] +
                '_F_N_' + str(nsp) + classicnorm +
                '.csv')).replace('\\', '/'), delimiter=',')
        elif opt.noise_output:
            far = torch.rand(nsp)
            f_n = torch.rand(opt.ampl)
        if not opt.noise_output and not opt.rand_angle and\
                enim == 1 and not opt.outputt == 'single_f_n':
            far = fara[int(angles_list[0]/10)]
            f_n = f_na[int(angles_list[0]/10)]
        path = os.path.join(self.df.iloc[idx, 0])
        if not opt.outputt == 'single_f_n':
            sample = {'image': img, 'f_n': f_n, 'far': far,\
                      'angles': angles_list, 'path': path}
        elif opt.outputt == 'single_f_n':
            f_n = np.genfromtxt(os.path.join(self.root_dir, (
                self.df.iloc[idx, 0] +
                '_F_N.csv')).replace('\\', '/'), delimiter=',')
#             print('img:',type(img), ' ', img.shape)
#             print('f_n:',type(f_n), ' ', f_n.shape)
#             print('angles:',angles_list)
#             print('path:',path)
            sample = {'image': img, 'f_n': f_n, 'angles': angles_list, 'path': path}
        if self.transform:
            sample = self.transform(sample)
        return sample

    
if not(opt.single_folder or opt.outputt == 'single_f_n'):
    exec(open(homepath+"circles/finetune_test/transform.py").read())
elif not opt.single_folder and opt.outputt == 'single_f_n':
    exec(open(homepath+"circles/finetune_test/transform_f_n.py").read())
elif opt.single_folder:
    exec(open("transform.py").read())


def shplot(coef, dirname, cnt):
    plt.rc('text', usetex=True)
    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([np.sin(theta) * np.sin(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta)])
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    Yx, Yy, Yz = plot_Y(ax, coef, xyz, phi, theta)
    plt.savefig(dirname + 'surface' + str(cnt).zfill(3) + '.png')
    cnt += 1
    return(cnt)


def plot_Y(ax, coef, xyz, phi, theta):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""
    f = np.zeros([100, 100]).astype('complex128')
    for od in range(int(np.sqrt(len(coef)))):
        for m in range(-od, od+1):
            fb = coef[od*(od+1)+m] * sph_harm(abs(m), od, phi, theta)
            f += fb
    Yx, Yy, Yz = np.abs(f) * xyz
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(f.real),
                    rstride=2, cstride=2)
#     print(Yz.shape)
    # Draw a set of x, y, z axes for reference.
    ax_lim = 50
    ax.plot([-ax_lim, ax_lim], [0, 0], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [-ax_lim, ax_lim], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [0, 0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax_lim = 40
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('on')
    return(Yx, Yy, Yz)

batch_size = opt.bs
if not opt.noise_output and opt.minmax:
    tmean = np.genfromtxt(
        os.path.join(homepath,'circles/finetune_test/tmean.csv'),
        delimiter=',')
else:
    tmean = np.zeros([5270, opt.ampl])
minmax, minmax3dimage, normalize, center, cmscrop, downsample = ['']*6
if opt.minmax:
    minmax = 'Minmax(tmean[:,:opt.ampl]), '
else:
    minmax = ''
if opt.inputt == 'pc' and opt.minmax3dimage:
    minmax3dimage = \
        'Minmax3Dimage(\
            np.array([29,  1,  4]), np.array([240, 138, 243])), '
else:
    minmax3dimage = ''
if opt.inputt == 'pc' and opt.normalize:
    normalize = 'Normalize(), '
else:
    normalize = ''
if opt.inputt == 'pc' and opt.center:
    center = 'Center(), '
else:
    center = ''
if not opt.inputt == 'pc' and opt.cmscrop:
    cmscrop = 'CmsCrop(opt.cmscrop),'
else:
    cmscrop = ''
if opt.inputt == 'pc' and opt.downsample:
    downsample = 'Downsample(opt.downsample), '
else:
    downsample = ''
if opt.inputt == 'f':
    minmax3dimage, normalize, center, cmscrop, downsample, rescale = ['']*6
if opt.inputt == 'f' and opt.minmax_f:
    minmax_f = 'Minmax_f((2.6897299652941, 102.121738007844)), '
#        minmax_f = 'Minmax_f((0.60117054538415,110.972068294924)), '
else:
    minmax_f = ''
if opt.inputt == 'img' and opt.rescale:
    rescale = 'Rescale(opt.rescale), '
else:
    rescale = ''
if opt.noise_input or opt.single_folder:
    (minmax, minmax3dimage, normalize, center,
     cmscrop, downsample, minmax_f) = ['']*7
    rescale = 'Rescale(opt.rescale), '
es = ("data_transforms = {\'train\': transforms.Compose([" +
     minmax + minmax3dimage + normalize + center +
     cmscrop + downsample + minmax_f + rescale +
     "AmpCrop(opt.ampl),ToTensor(device)]),\
     'val': transforms.Compose([" +
     minmax + minmax3dimage + normalize + center +
     cmscrop + downsample + minmax_f + rescale +
     "AmpCrop(opt.ampl),ToTensor(device)])}")
# print(es)
exec(es)
# kwargs = {}
# kwargs = {'num_workers': 1, 'pin_memory': True} if opt.parallel == 'hvd' else {}
kwargs = {'num_workers': opt.num_workers, 'pin_memory': opt.pin_memory} if opt.parallel == 'hvd' else {}
# if (opt.parallel == 'hvd' and
#     kwargs.get('num_workers', 0) > 0 and
#     hasattr(mp, '_supports_context') and mp._supports_context and
#         'forkserver' in mp.get_all_start_methods()):
#     kwargs['multiprocessing_context'] = 'forkserver'

#print("Initializing Datasets and Dataloaders...")

excpath = os.path.join(mainpath,opt.specie+'_exceptions.txt')
if not os.path.isfile(excpath) and not opt.single_folder:
    cip = []
    # i = 0
    for root, directories, filenames in os.walk(os.path.join(mainpath, opt.csvname)):
        for filename in filenames:
            if 'F_N.csv' in filename:
    #             F_Nm[i,:] = np.genfromtxt(
    #                 os.path.join(root,filename),delimiter=',')
    #             i+=1
                cip.append(os.path.join(root,filename))
    F_Nm = np.zeros([len(cip),opt.ampl])
    for i in range(len(cip)):
        F_Nm[i,:] = np.genfromtxt(cip[i],delimiter=',')
    for i in range(opt.ampl):
        ar = F_Nm[:,i]
        a = np.concatenate((a,np.where(np.abs(ar-np.mean(ar)) > 6*np.std(ar))[0]))
    exception_list = [cip[index][-23:-8] for index in np.unique(a.astype(int))]
    with open(excpath, "w") as f:
        for element in exception_list:
            f.write(element + "\n")
elif os.path.isfile(excpath) and not opt.single_folder:
    with open(excpath, "r") as f:
        exception_list = f.readlines()
else:
    exception_list = []
pts = os.path.join(mainpath, 'pathes_to: ' + opt.specie + '.csv')
# print(pts)
if opt.noise_output:
    lframe = pd.DataFrame()
    lframe.insert(
        0, 'file_name', np.zeros(5270))
elif not opt.noise_output and os.path.isfile(pts) and opt.use_existing_csv:
    lframe = pd.read_csv(pts)
elif not opt.noise_output and not(os.path.isfile(pts) or opt.use_existing_csv):
    cip = []
    for word in opt.specie.split(','):
        csvpath = os.path.join(mainpath, word+'csv')
        for root, directories, filenames in os.walk(csvpath):
            for filename in filenames:
                search_str = 'F_N_' + str(nsp) + '.csv'
                for i in range(len(exception_list)):
                    if not any(exception in cip[i] for exception in exception_list):
#                 if search_str in filename and flag:
                        tfolds = os.path.join(root, filename).split('/')[-3:]
                        cip.append(os.path.join(tfolds[0], tfolds[1], tfolds[2]))
    lframe = pd.DataFrame()
    lframe.insert(
        0, 'file_name', [cip[i].split('_F_N')[0] for i in range(len(cip))])

    list_zero = []
    for idx in range(len(lframe)):
        s = np.zeros(36)
        rotpath = os.path.join(
            mainpath.replace('phenoseed_csv', 'phenoseed'),
            lframe.iloc[idx, 0].replace('csv', ''))
        for j in range(36):
            img = io.imread(
                os.path.join(
                    rotpath, 'rotation_' + str(10*j).zfill(3) + '.tif'))
            s[j] = np.sum(255-img)
        list_zero.append(np.argmax(s))
    lframe.insert(1, 'zero_angle', list_zero)
    lframe = lframe.sample(frac=1)
    lframe.to_csv(pts, index=False)
lframe = lframe.replace({opt.specie+'csv':opt.csvname}, regex=True)
inn = opt.inputt == 'img' and not opt.noise_input
if inn and not opt.rand_angle:
    alf = np.array(
        [10*int(36*i/nim) for i in range(nim)])
elif inn and opt.rand_angle:
    alf = np.random.choice(
        np.arange(0, 360, 10), size=nim, replace=False)
elif inn and nim == 1 and opt.zero_angle:
    alf = [0]
pts = os.path.join(mainpath, opt.specie + '_view_sep.csv')
if opt.view_sep and not(os.path.isfile(pts) and\
         opt.use_existing_csv):
    lframe_sep = pd.DataFrame(columns=('file_name', 'angle'))
    for idx in range(len(lframe)):
        for j, angle in enumerate(alf):
            if lframe.iloc[idx, 1]+int(angle/10) > 36:
                angle = lframe.iloc[idx, 1]+int(angle/10)-36
            else:
                angle = lframe.iloc[idx, 1]+int(angle/10)
            lframe_sep.loc[idx*len(alf)+j] = [lframe.iloc[idx, 0], angle]
    lframe = lframe_sep
    lframe.to_csv(pts, index=False)
elif opt.view_sep and os.path.isfile(pts) and opt.use_existing_csv:
    lframe = pd.read_csv(pts)
for st in exception_list:
    lframe = lframe[~lframe.file_name.str.contains(st)]
if opt.single_folder:
    lframe = lframe[:100]
train_part = int(0.8*len(lframe))
df_dict = {'train': lframe[:train_part], 'val': lframe[train_part:]}
# Create training and validation datasets
image_datasets = {x: Seed3D_Dataset(
    mainpath, df_dict[x],
    transform=data_transforms[x]) for x in ['train', 'val']}

if opt.parallel == 'hvd':
    samplers = {x: torch.utils.data.distributed.DistributedSampler(
        image_datasets[x], num_replicas=hvd.size(),
        rank=hvd.rank(), shuffle=False) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=opt.bs, shuffle=False, sampler=samplers[x],
        **kwargs) for x in ['train', 'val']}
elif opt.parallel == 'torch':
    samplers = None
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)
        for x in ['train', 'val']}


# # Gather the parameters to be optimized/updated in this run. If we are
# #  finetuning we will be updating all parameters. However, if we are
# #  doing feature extract method, we will only update the parameters
# #  that we have just initialized, i.e. the parameters with requires_grad
# #  is True.
# params_to_update = model_ft.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)

criterion = nn.MSELoss(reduction='mean')
bn = opt.bn
num_epochs = opt.epoch
lossar = np.zeros([2, num_epochs])
# Train and evaluate
exec(opt.hidden_dim)
exec(opt.chidden_dim)
exec(opt.kernel_sizes)

if bool(opt.model_name):
    smodel, input_size = initialize_model(
        opt.model_name, opt.ampl, opt.feature_extract,
        use_pretrained=opt.use_pretrained)
elif not bool(opt.model_name) and opt.inputt == 'img':
    smodel = CNet(
        hidden_dim, chidden_dim, kernel_sizes,
        nim, opt.rescale,
        int(opt.rescale*original_w/original_h), opt.haf)
elif not bool(opt.model_name) and opt.inputt == 'f':
    smodel = CNet(
        hidden_dim, chidden_dim, kernel_sizes,
        nim, 1, nsp, opt.haf)
elif not bool(opt.model_name) and opt.inputt == 'pc':
    smodel = CNet(
        hidden_dim, chidden_dim, kernel_sizes,
        nim, np.floor(58014/opt.downsample).astype(int),
        1, opt.haf)
smodel = nn.DataParallel(smodel) if opt.parallel == 'torch' else smodel

lr = opt.lr*hvd.local_size() if opt.parallel == 'hvd' else opt.lr
smodel.cuda()
optimizer = torch.optim.Adam(
    smodel.parameters(), lr, betas=(0.9, 0.999), eps=1e-08,
    weight_decay=opt.weight_decay, amsgrad=False)

if opt.parallel == 'hvd':
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(smodel.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.none
    # compression = hvd.Compression.fp16 if
    # args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=smodel.named_parameters(),
        compression=compression,
        op=hvd.Adasum if opt.use_adasum else hvd.Average,
        gradient_predivide_factor=opt.gradient_predivide_factor)
model, lossar, time_elapsed, dirname = train_model(
    smodel, dataloaders_dict, samplers, criterion, optimizer, tmean,
    mainpath, num_epochs=num_epochs,
    is_inception=(opt.model_name == "inception"))

if rank == 0 and opt.save_output:
    with open(dirname+"job-parameters.txt", 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f  # Change the standard output to the file we created.
        print(smodel)
        # Reset the standard output to its original value
        sys.stdout = original_stdout
        f.write('time_elapsed=' +\
                str(datetime.timedelta(seconds=time_elapsed)) + '\n')
#     phase = 'val'
#     bn = 1
#     dataloaders = dataloaders_dict
#     for i_batch, sample_batched in enumerate(dataloaders[phase]):
#         if i_batch == bn:
#             break
#         inputs = sample_batched['image']
#         if opt.inputt:
#             x = inputs
#             y = torch.zeros([x.shape[0], x.shape[1]])
#             y = y.cuda()
#             for i in range(x.shape[0]):
#                 y[i, :] = torch.sqrt(
#                     x[i, :, 0]**2 + x[i, :, 1]**2 + x[i, :, 2]**2)
#             y = torch.unsqueeze(y, 2)
#             o = model(y)
#         else:
#             o = model(inputs)
#         gt = sample_batched['landmarks']

#     if opt.minmax:
#         ampl = opt.ampl
#         output = np.multiply(model(inputs).detach().cpu().numpy(),
#                              tmean[3, :ampl] -
#                              tmean[2, :ampl]) + tmean[2, :ampl]
#         real_output = np.multiply(gt.detach().cpu().numpy(),
#                                   tmean[3, :ampl] -
#                                   tmean[2, :ampl]) + tmean[2, :ampl]
#     else:
#         output = o.detach().cpu().numpy()
#         real_output = gt.detach().cpu().numpy()

    torch.save(model.state_dict(), dirname+"model")
    for x in ['train', 'val']:
        df_dict[x].to_csv(dirname+'pathes_df_'+x+'.csv')
    print('ellapsed time = ', time.time()-tstart)
    # ## Return SH coefficient vector from the trained model