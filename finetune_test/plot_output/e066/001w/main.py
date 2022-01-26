#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from __future__ import division
import argparse
import open3d
import torch as t
# here was hvd import
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

import torch.multiprocessing as mp
import shutil
import imageio
from inspect import currentframe, getframeinfo
import h5py
from functools import wraps
from pathlib import Path
from os.path import join as jn
from seed_everything import seed_everything
import random
from torch.optim.lr_scheduler import StepLR
from helpers import *
import csv
# from torch import linalg as talg
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def setarg(parser, argname, dfl):
        parser.add_argument('-'+argname, dest=argname,
                            action='store_true')
        parser.add_argument('-no_'+argname, dest=argname,
                            action='store_false')
        exec('parser.set_defaults('+argname+'=dfl)')
    parser.add_argument('-bs', type=int, default=4)
    parser.add_argument('-epoch', type=int, default=8)
    # parser.add_argument('-bn', type=int, default=8)
    parser.add_argument('-lr', type=float, default=5e-5)
    # normalize output on the tmean matrix, to have min = 0 and max = 1
    setarg(parser, 'minmax',False)
    # normalize input point cloud to have every coordinate between 0 and 1
    setarg(parser, 'minmax3dimage',False)
    #normalize input point cloud, that it is in canonical view
    setarg(parser, 'normalize',False)
    #centerize input point cloud, to have it's center of masses in the origin
    setarg(parser, 'center',False)
    #linearly downsample input point cloud
    parser.add_argument('-downsample', type=int, default=1)

    #use f_n or f, that was gotten with normalization on canonical view before processing
    setarg(parser, 'classicnorm',False)

    #cut the number of maximum SH amplitude to regress
    parser.add_argument('-ampl', type=int, default=441)

    #centerize seed on the input image and crop to this width
    parser.add_argument('-cmscrop', type=int, default=0)
    parser.add_argument('-cencrop', type=int, default=800)
    #rescale input image
    parser.add_argument('-rescale', type=int, default=250)
    setarg(parser, 'use_adasum',False)
    parser.add_argument(
        '-gradient_predivide_factor', type=float, default=1.0,
        help='apply gradient predivide factor in optimizer (default: 1.0)')

    #name of experiment directory
    parser.add_argument('-expnum', type=str, default='111')
    # parser.add_argument('-hidden_dim',type=str,
    #   default = 'hidden_dim = np.hstack((np.repeat(128, 3),441))')

    # hidden_dim - size of appendix FC layers
    parser.add_argument(
        '-hidden_dim', nargs='+', type=int, default=[5000,2500,1000,441])
    # parser.add_argument(
    #     '-hidden_dim', type=str,
    #     default='hidden_dim=np.array([5000,2500,1000,441])')
    # parser.add_argument(
    #     '-chidden_dim', type=str,
    #     default='chidden_dim = np.hstack((96,128,np.repeat(256, 3)))')
    parser.add_argument(
        '-chidden_dim', nargs='+', type=int, default=[96, 128, 256, 256, 256])
    # parser.add_argument(
    #     '-kernel_sizes', type=str,
    #     default='kernel_sizes = np.hstack((7,np.repeat(3, 5)))')
    parser.add_argument('-kernel_sizes', nargs='+',  default=[7, 3, 3, 3, 3, 3])
    #number of input images that will be loaded
    parser.add_argument('-num_input_images', type=int, default=1)
    #name of standard model
    parser.add_argument('-model_name', type=str, default='')
    parser.add_argument('-netname', nargs='+', default=['cnet'])
    # parser.add_argument('-netname', nargs='+', default=['cnet','decoder','encoder'])
    setarg(parser, 'use_pretrained',False)
    parser.add_argument('-weight_decay', type=float, default=0)

    #used to load images all in parallel, or merge them after output
    parser.add_argument('-merging_order', type=str,
        choices=['color_channel', 'latent_merge'], default='latent_merge')
    #take input image of random angle, if not, then image will
    #be taken relative to the horizontal pose
    setarg(parser, 'rand_angle',False)
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
    setarg(parser, 'use_existing_csv',True)
    setarg(parser, 'use_sep_csv',True)
    #instead of input files noise is generating with random numbers
    setarg(parser, 'noise_input',False)
    #use convolutional part of the network or not
    setarg(parser, 'haf',True)
    #type of input data. can be 'img', 'f' or 'pc'
    parser.add_argument('-inputt', type=str, default='img')
    #normalize to make min = 0 and max = 1 for input f
    setarg(parser, 'minmax_f',True)
    #criterion to calculate loss
    parser.add_argument('-criterion', type=str, default='L1')
    #number of GPUs is used in the job
    parser.add_argument('-ngpu', type=int, default=4)
    #type of parallelization. 'hvd' means horovod, or 't'
    parser.add_argument('-parallel', type=str, default='hvd')
    #in case loading standard model, it can be use as feature extracting
    #(when freezeing all layers except the last one)
    setarg(parser, 'feature_extract',False)
    #if load only one image as input, this will be always image with index 000_rotation
    #if load more than 1 image, then number of images will be spread evenly in
    #the range (0,36)
    #if false, images will be taking that first image in views will be with
    #horizontal pose
    setarg(parser, 'zero_angle',False)
    #is used for testing computing time,
    #where all needed files including data in one folder
    parser.add_argument('-single_folder',
                        dest='single_folder', action='store_true')
    parser.set_defaults(single_folder=False)

    parser.add_argument('-noise_output', dest='noise_output',
                        action='store_true')
    parser.set_defaults(noise_output=False)
    #only log will be in the output
    setarg(parser, 'save_output',True)
    #type of data that is loaded for gt. for example, single_f_n
    #means that only *f_n files will be used for GT in dataloader
    #and maybe it will be singular loading of y_n
    #it is used separate transform_f_n.py to not load more than is
    #needed
    parser.add_argument('-gttype', type=str, default='single_f_n')
    #name of csv that will be used for loading GT
    parser.add_argument('-csvname', type=str, default='598csv9')
    #factor on which all output point cloud data will be normalized
    parser.add_argument('-pscale', type=int, default=100)

    #if view_sep = True, and more than one image is loaded,
    # all input images will be treated as separate data elemets
    setarg(parser, 'view_sep',False)
    #rotate directions together with angle from which
    #current image were taken
    setarg(parser, 'rot_dirs',False)
    #for dataloader
    parser.add_argument('-num_workers', type=int, default=0)
    setarg(parser, 'pin_memory',False)
    #manually calculate distance vector F out of point cloud output
    setarg(parser, 'man_dist',False)
    setarg(parser, 'use_cuda',True)
    parser.add_argument('-machine', type=str,
        choices=['jureca', 'workstation', 'lenovo', 'huawei'], default='jureca')
    setarg(parser, 'maintain',False)
    setarg(parser, 'maintain_line',False)
    parser.add_argument('-wandb', type=str, default="")
    setarg(parser, 'measure_time',False)
    setarg(parser, 'rotate_output',False)
    parser.add_argument('-transappendix', type=str, default="_image")
    parser.add_argument('-batch_output', type=int, default=50)
    parser.add_argument('-minmax_fn', type=str, default='')
    parser.add_argument('-updateFraction', type=int, default=3)
    parser.add_argument('-standardize',  nargs='+', default=None)
    # parser.add_argument('-standardize', default=(18.31589541, 39.63290785))
    parser.add_argument('-localexp', type=str, default='')
    parser.add_argument('-steplr',  nargs='+', type=float, default=(30,1))
    parser.add_argument('-outputt', type=str, choices=['points','pose6'], default='points')
    parser.add_argument('-ufmodel', type=int, default=100000)
    parser.add_argument('-framelim', type=int, default=int(1e20))
    parser.add_argument('-conTrain', type=str, default='')
    # print('hello from 201')
    opt = parser.parse_args()
    # print(opt, len(opt), opt[0])
    # print('opt dict', opt.__dict__)
    if opt.parallel == 'hvd':
        import horovod.t as hvd
        hvd.init()
        t.cuda.set_device(hvd.local_rank())
        t.set_num_threads(1)
        rank = hvd.rank()
    else:
        rank = 0
    # print('215')
    if opt.machine == 'jureca':
        homepath = '/p/home/jusers/cherepashkin1/jureca/'
    elif opt.machine == 'workstation' or opt.machine == 'huawei':
        homepath = 'D:/'
    elif opt.machine == 'lenovo':
        homepath = 'C:/'
    dir1 = jn(homepath, 'cherepashkin1', '598test', 'plot_output',opt.expnum)
    if opt.save_output and rank == 0 and not opt.localexp:
        dirname = None
        while dirname is None:
            try:
                dirname = newfold(dir1, opt.machine[0])
            except:
                time.sleep(1)
                pass
        # print('224')
    elif opt.localexp and opt.save_output and rank == 0:
        dirname = jn(dir1, opt.localexp)
        for filename in os.listdir(dirname):
            file_path = jn(dirname, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        dirname = jn(dir1, 'misc')
    # print('236')
    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)
    def dic2csv(path, dct):
        with open(path, 'w', newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for key, value in dct.items():
                writer.writerow([key, value])
    def csv2dic(path):
        with open(path, 'r', newline='\n') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            return Bunch(dict(reader))
    # print('bs line 1382', opt.bs)
    conTrain = opt.conTrain
    if opt.conTrain:
        # opt = None
        print(jn(dir1,conTrain,'opt.csv'))
        opt = csv2dic(jn(dir1,conTrain,'opt.csv'))
    dic2csv(jn(dirname,'opt.csv'), opt.__dict__)

    # opt.bs = 30
    # print('opt bs =,', opt.bs)

    # print(opt.use_existing_csv)
    # print(opt.use_sep_csv)
    # print(get_ipython().__class__.__name__)
    nim = opt.num_input_images
    enim = nim if not opt.view_sep else 1
    nsp = opt.num_sam_points
    tstart = time.time()
    iscuda = opt.use_cuda and t.cuda.is_available()
    # print(opt.use_cuda, t.cuda.is_available())
    print('iscuda=',iscuda)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    seed = 0
    if iscuda:
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False
    # random.seed(seed)
    # np.random.seed(seed)
    # t.manual_seed(seed)
    #t.use_deterministic_algorithms(True)
    seed_everything(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    def worker_init_fn(worker_id):
        worker_seed = t.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = t.Generator()
    g.manual_seed(seed)
    def generator(seed):
        g = t.Generator()
        g.manual_seed(seed)
        return g
    vox2mm = 0.05
    # opt.pscale *= vox2mm

    mt = opt.maintain and rank == 0
    mt1 = opt.maintain_line and rank == 0
    if rank == 0:
        print("PyTorch Version: ", t.__version__)
        print("Torchvision Version: ", torchvision.__version__)
        print('opt:\n',opt)
        print('sys.argv:\n',sys.argv)
        print('seed = ', seed)
    classicnorm = '_prenormalized' if opt.classicnorm else ''
    if '619' in opt.specie:
        original_h, original_w = 2048, 2448
    else:
        original_h, original_w = 1000, 1800

    # if not opt.laptop:
    # else:
    #     homepath = 'D:/'
    #path1 = jn(homepath, 'circles/finetune_test')
    finePath = jn(homepath, 'circles', 'finetune_test')
    dataPath = jn(homepath, 'cherepashkin1', 'phenoseed', opt.specie)
    csvPath = jn(finePath, 'csv')
    # path1 = os.path.abspath(__name__).replace('__main__','')
    if rank == 0:
        print('path were main.py is located=',finePath)
        print('homepath is ', homepath)
    #os.chdir(path1)
    #print(os.getcwd())
    #print(os.path.abspath(__name__).replace('__main__',''))
    if opt.single_folder:
        exec('from'+opt.netname+'import *')
    else:
    #     exec(open(jn(['/p/home/jusers/cherepashkin1/',
    #                             'jureca/cherepashkin1/experiments'],
    #                             opt.expnum+'cnet.py')).read())
        for name in opt.netname:
            exec('from experiments.'+opt.expnum+'.'+name+' import *')
    def my_loss(output, target):
        myloss = t.mean(t.multiply(weightv, (output - target))**2)
        return myloss
    # maintainl = 'if mt1:\n\t frameinfo = getframeinfo(currentframe())\n\t'\
    #              'print(frameinfo.filename, frameinfo.lineno)'
    # def mtl(mt1):
    #     if mt1:
    #         frameinfo = getframeinfo(currentframe())
    #         print(frameinfo.filename, frameinfo.lineno)
    # # mtl(mt1)
    # if mt1:
    #     frameinfo = getframeinfo(currentframe())
    #     print(frameinfo.filename, frameinfo.lineno)
    # if not opt.single_folder:
    #     mainpath = jn(finePath,'csv')
    #     # mainpath = jn(homepath, 'cherepashkin1/phenoseed_csv')
    # elif opt.single_folder and opt.machine == 'jureca':
    # #     print('__file__', __file__)
    # #     mainpath = jn(os.path.abspath(__file__).replace('smain.py',''),'phenoseed_csv')
    #     mainpath = jn(homepath, 'seed2sh')
    # elif opt.laptop:
    #     mainpath = 'D:\data\seeds\phenoseed_csv'



    if opt.save_output and rank == 0 and not os.path.exists(dir1):
        os.mkdir(dir1)
    # print('dir1', dir1)
    # print('path1', path1)
    sys.path.insert(1, finePath)
    sys.path.insert(2, dir1)

    if opt.model_name:
        from standard_models import *


    # def forwardproj(pcd, pr):
    #     if isinstance(pcd, t.Tensor) and isinstance(pr, np.ndarray):
    #         pr = t.Tensor(pr)
    #         pr = pr.cuda() if iscuda else pr
    #     if isinstance(pcd, t.Tensor) and isinstance(pr, np.ndarray):
    #         pcd2 = t.cat((pcd, np.expand_dims(
    #             np.repeat(1, pcd.shape[0]), axis=1)), axis=1)
    #         pcd2 = np.matmul(pcd2, pr.T)
    #         pcd2 = pcd2 / np.repeat(np.expand_dims(pcd2[:, 3], axis=1), 4, axis=1)
    #     if isinstance(pcd, np.ndarray) and isinstance(pr, t.Tensor):
    #         pr = pr.detach().cpu().numpy())
    #     if isinstance(pcd, np.ndarray) and isinstance(pr, np.ndarray):
    #         pcd2 = np.concatenate((pcd, np.expand_dims(
    #             np.repeat(1, pcd.shape[0]), axis=1)), axis=1)
    #         pcd2 = np.matmul(pcd2, pr.T)
    #         pcd2 = pcd2 / np.repeat(np.expand_dims(pcd2[:, 3], axis=1), 4, axis=1)
    #
    #     return (pcd2[:, :3])


    # parameters_list = (
    #     'expdescr: ' + opt.expdescr + '\n' +
    #     'epoch='+str(opt.epoch) + '\n' +
    #     'bn='+str(opt.bn) + '\n' +
    #     'bs='+str(opt.bs) + '\n' +
    #     'lr='+str(opt.lr) + '\n' +
    #     'minmax=' + str(opt.minmax) + '\n' +
    #     'minmax3dimage=' + str(opt.minmax3dimage) + '\n' +
    #     'downsample=' + str(opt.downsample) + '\n' +
    #     'normalize=' + str(opt.normalize) + '\n' +
    #     'center=' + str(opt.center) + '\n' +
    #     'classicnorm=' + str(opt.classicnorm) + '\n' +
    #     'ampl=' + str(opt.ampl) + '\n' +
    #     'cmscrop=' + str(opt.cmscrop) + '\n' +
    #     'rescale=' + str(opt.rescale) + '\n' +
    #     'use_adasum=' + str(opt.use_adasum) + '\n' +
    #     'gradient_predivide_factor=' +
    #     str(opt.gradient_predivide_factor) + '\n' +
    #     'expnum=' + str(opt.expnum) + '\n' +
    #     'hidden_dim=' + opt.hidden_dim + '\n' +
    #     'chidden_dim=' + opt.chidden_dim + '\n' +
    #     'kernel_sizes=' + opt.kernel_sizes + '\n' +
    #     'inputt == pc, img_input, F input? =' +
    #     str(opt.inputt == 'pc') + str(opt.inputt == 'img') +
    #     str(opt.inputt == 'f') + '\n' +
    #     'model_name=' + opt.model_name + '\n' +
    #     'feature_extract=' + str(opt.feature_extract) + '\n' +
    #     'Was it used pretrained model ? =' +
    #     str(opt.use_pretrained) + '\n' +
    #     'weight_decay=' + str(opt.weight_decay) + '\n' +
    #     'num_input_images=' + str(nim) + '\n' +
    #     'merging_order=' + str(opt.merging_order) + '\n' +
    #     'specie=' + opt.specie + '\n' +
    #     'number of sampled points=' + str(nsp) + '\n' +
    #     'loss is between model output and gt of =' +
    #     opt.lb + '\n' +
    #     'rand_angle=' + str(opt.rand_angle) + '\n' +
    #     'zero_angle=' + str(opt.zero_angle) + '\n' +
    #     'ngpu=' + str(opt.ngpu) + '\n' +
    #     'gttype=' + opt.gttype + '\n' +
    #     'csvname=' + opt.csvname + '\n' +
    #     'pscale=' + str(opt.pscale) + '\n' +
    #     'view_sep=' + str(opt.view_sep) + '\n' +
    #     'rot_dirs=' + str(opt.rot_dirs) + '\n' +
    #     'was there just noise as input? =' + str(opt.noise_input))

    print('opt.wandb = ',opt.wandb)

    if opt.wandb:
        import wandb
    if opt.wandb != 'disabled' and opt.wandb and opt.machine != 'jureca':
        wandb.init(project=opt.wandb, config=vars(opt))
    elif opt.wandb != 'disabled' and opt.wandb and opt.machine == 'jureca':
        wandb.init(mode="offline", config=vars(opt))
    elif opt.wandb == 'disabled':
        wandb.init(mode="disabled")
    def savelist(listf, filename):
        with open(filename, "w") as f:
            for element in listf:
                f.write(element + "\n")
    lb = opt.lb
    epoch0 = -1
    # from time import time
    # from functools import wraps


    #@simple_time_tracker(_log)
    class Prepare_train(object):
        def __init__(self, **argd):
            self.__dict__.update(argd)

    def trainit(lframe, train_part, bs, nsp, classicnorm, csvname, ampl,
                rank, finePath, rot_dirs, merging_order, iscuda):
        val_acc_history = []
        # dataloaders =

        if rank == 0:
            print(time.ctime()) # 'Mon Oct 18 13:35:29 2010'
            # print('dataset is using %d views' %int((len(dataloaders['train'])+\
            #     len(dataloaders['val']))*opt.bs/5270))
            lastbs = []
            lbs = []
            ebn = []
            # 0.8*len(lframe)
            trFrac = [0.8, 1-0.8]
            lfl = [train_part, len(lframe)-train_part]
            # numOfBatches = len(lframe)//bs
            for pcnt, phase in enumerate(['train', 'val']):
                lbs.append(lfl[pcnt]%bs if lfl[pcnt]%bs else bs)
                ebn.append(int(len(lframe)*trFrac[pcnt]/bs))
                # if lfl[pcnt]%opt.bs else
                # len(dataloaders[phase]))
                print('{} consists of {} full batches with {} tensors with {}'\
                ' views'.format(phase, ebn[pcnt], opt.bs, nim))
                if lfl[pcnt]%opt.bs:
                    print('the last batch has size of {}'\
                    ' tensors with {} views'.format(lbs[pcnt], nim))
        # print(classicnorm)
        # y_n = np.genfromtxt(jn(
        #     csvPath, 'y_n_fibo_' +
        #     str(nsp) +
        #     classicnorm+'whole.csv').replace('\\', '/'),
        #     delimiter=',')
        # bX = np.genfromtxt(jn(
        #     csvPath, 'bX_fibo_' +
        #     str(nsp) +
        #     classicnorm+'whole.csv').replace('\\', '/'),
        #     delimiter=',')
        # F_Nw = np.array(h5py.File(jn(
        #     csvPath, csvname+'_F_N.h5').replace('\\', '/'),
        #         'r').get('dataset'))
        # prmatw = np.array(h5py.File(jn(
        #     csvPath, opt.specie+'prmat.h5').replace('\\', '/'),
        #         'r').get('dataset'))
        y_n, bX, F_Nw, prmatw = [np.array(h5py.File(jn(csvPath,nm), 'r').get('dataset')) for nm in
                                 ['y_n_fibo_' + str(nsp) + classicnorm+'_whole.h5',
                                  'bX_fibo_' + str(nsp) + classicnorm+'_whole.h5',
                                  csvname+'_F_N.h5',
                                  opt.specie+'prmat.h5']]
        bigm = np.copy(prmatw)
        bigm[:, :, 3, :] = np.repeat(np.repeat(np.expand_dims([0, 0, 0, 1],
            axis=(0, 1)), 36, axis=1), prmatw.shape[0], axis=0)
        matw = np.einsum('ij, njk->nik', np.linalg.inv(bigm[0, 0, :, :]), bigm[:, 0, :, :])
        # basism = np.array([[740.39063, 0, -45.703125, -182807.09],
        #             [0, 740.39063, -27.21875, -99071],
        #             [0, 0, -0.05078125, -121],
        #             [0, 0, 0, 1]])
        # basism = bigm[0,:,:,:]
        # basisminv = np.linalg.inv(basism)
        F_Nw = F_Nw[:,:ampl]
        tmean = np.zeros([2,F_Nw.shape[0],ampl])
        # print('line 449')

        # dirname = dirname[:-1]+opt.machine[0]
        # print('line 458')

        for x in ['train', 'val']:
            df_dict[x].to_csv(jn(dirname,'pathes_df_'+x+'.csv'), index=False)
        # lframe[:train_part].to_csv(dirname+'lframe_train', index=False)
        # lframe[train_part:].to_csv(dirname+'lframe_val', index=False)
        # def tmeanget(F_N,fun):
        #     exec('tmeanc = np.repeat(np.expand_dims(np.'+\
        #         fun+'(F_N,0),axis=0),F_N.shape[0],axis=0)')
        #     return(tmeanc)
        # for i in range(2):
        #     tmean[i] = tmeanget(F_Nw,opt.minmax_fn.split(',')[i])
        if opt.minmax_fn == 'min,max':
            tmean[0] = np.repeat(np.expand_dims(np.nanmin(F_Nw,0),
                axis=0),F_Nw.shape[0],axis=0)
            tmean[1] = np.repeat(np.expand_dims(np.nanmax(F_Nw,0),
                axis=0),F_Nw.shape[0],axis=0)
            F_Nw = (F_Nw - tmean[0])/(tmean[1]-tmean[0])
        elif opt.minmax_fn == 'mean,std':
            tmean[0] = np.repeat(np.expand_dims(np.nanmean(F_Nw,0),
                axis=0),F_Nw.shape[0],axis=0)
            tmean[1] = np.repeat(np.expand_dims(np.nanstd(F_Nw,0),
                axis=0),F_Nw.shape[0],axis=0)
            F_Nw = (F_Nw - tmean[0])/tmean[1]
        # F_N = pd.read_hdf(jn(
        #     mainpath, 'F_N_' +
        #     opt.csvname+'.h5').replace('\\', '/'), key='df', mode = 'r')
        # F_N = np.genfromtxt(jn(
        #     mainpath, 'F_N_' +
        #     opt.csvname).replace('\\', '/'),
        #     delimiter=',')
        prmat = np.genfromtxt(jn(
            finePath, 'csv', 'prmat.csv').replace('\\', '/'),
            delimiter=',')
        C = np.zeros([36,3,3])
        E = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0]]
        # TODO exchange loop with vectorization with einsum
        for i in range(36):
            C[i,:,:] =\
            np.matmul(np.matmul(E,prmat[4*i:4*(i+1),:]),
                np.linalg.pinv(np.matmul(E,prmat[0:4,:])))
        bX = t.transpose(t.Tensor(bX),0,1)
        C, y_n, F_Nw, prmatw, bigm, matw = [t.Tensor(i) for i in\
                                            [C, y_n, F_Nw, prmatw, bigm, matw]]
        if iscuda:
            bX, C, y_n, F_Nw, prmatw, bigm, matw = [i.cuda() for i in\
                [bX, C, y_n, F_Nw, prmatw, bigm, matw]]
        tmeant = t.Tensor(tmean).cuda() if iscuda else t.Tensor(tmean)

        if (lb == 'f' or lb == 'pc' or lb == 'pc+f') and not rot_dirs:
            # y_n2=[]
            # dirs=[]
            y_n2_one = t.unsqueeze(y_n[:nsp,:], axis=0)
            # y_n2_one = t.Tensor(y_n2_one).cuda() if iscuda else\
            #     t.Tensor(y_n2_one)
            # dirs = t.Tensor(dirs).cuda() if iscuda else t.Tensor(dirs)
            y_n2 = y_n2_one.repeat(opt.bs*enim, 1, 1)
            dirs = bX[:, :2].repeat(opt.bs*enim, 1, 1)
            # for pcnt in range(2):
            #     y_n2.append(y_n2_one.repeat(lastbs[pcnt], 1, 1))
            #     dirs.append(bX[:, :2].repeat(lastbs[pcnt], 1, 1))
        if all([not rot_dirs, merging_order == 'color_channel']):
            y_n2_one = t.unsqueeze(y_n[:nsp,:], axis=0)
            # print(y_n2_one.shape)
            y_n2 = y_n2_one.expand(bs, nsp, ampl)
            dirs = bX[:, :2].expand(bs, nsp, 2)
        # print(bX.shape)
        return y_n, bX, F_Nw, tmean, bX, C, y_n, y_n2, dirs, \
               prmatw, bigm, matw

    def train_model(model, optimizer):
        # print('ttrain consists of %d batches with size' %len(dataloaders['train']))

        since = time.time()

        gt0 = []
        gt = []
        lossar = np.zeros([4, opt.epoch])
        lossarb = np.zeros([2, opt.epoch])
        curloss = np.zeros([2, opt.epoch])
        # lt = np.zeros([4, (len(lframe)*trFrac[0]//bs+1)])


        cnt = 0
        y_n, bX, F_Nw, tmean, bX, C, y_n, y_n2, dirs, prmatw,\
        bigm, matw =\
            trainit(lframe, train_part, opt.bs,
                    nsp, classicnorm, opt.csvname, opt.ampl,
                    rank, finePath, opt.rot_dirs,
                    opt.merging_order, iscuda)
        for epoch in range(opt.epoch):
            if opt.parallel == 'hvd':
                samplers = {x: t.utils.data.distributed.DistributedSampler(
                    image_datasets[x], num_replicas=hvd.size(),
                    rank=hvd.rank(), shuffle=False) for x in ['train', 'val']}
                # Create training and validation dataloaders
                dataloaders = {x: t.utils.data.DataLoader(
                    image_datasets[x],
                    batch_size=opt.bs, shuffle=False, sampler=samplers[x],
                    worker_init_fn=seed_worker,
                    generator=g, **kwargs) for x in ['train', 'val']}
            elif opt.parallel == 'torch':
                samplers = None
                dataloaders = {'train': t.utils.data.DataLoader(
                    image_datasets['train'],
                    batch_size=batch_size, shuffle=True, num_workers=opt.num_workers),
                    'val': t.utils.data.DataLoader(
                        image_datasets['val'],
                        batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)
                        }
            if mt:
                print('start training', time.time())
            if rank == 0:
                print('Epoch {}/{}'.format(epoch, opt.epoch - 1))
                print('-' * 10)
                print(time.ctime())
            ste = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    pcnt = 0
                    model.train()  # Set model to training mode
                else:
                    pcnt = 1
                    model.eval()   # Set model to evaluate mode
                rloss = 0.0
                rloss0 = 0.0
                rloss1 = 0.0
                if opt.parallel == 'hvd':
                    train_sampler = samplers[phase]
                    train_sampler.set_epoch(epoch)
                # Iterate over data.
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    # print(idx)
                    bst = time.time()
                    if mt:
                        print('start %d batch loading at %f' %(i_batch, time.time()))
                    if opt.measure_time:
                        ts = time.time()
                    # if i_batch == bn:
                    #     break
                    # pathes = sample_batched['path']
                    # print(pathes)
                    # print(sample_batched[0])
                    inputs = sample_batched[0]['image']
                    t2int = lambda x : [int(x[i]) for i in range(len(x))]
                    index = t2int(sample_batched[1].tolist())
                    # print('idx,', idx)
                    angles_list = sample_batched[2]
                    # idx = t2int(idx)
                    # idx = [int(idx[i]) for i in range(len(idx))]
                    # print('idx', idx)
                    pathes = lframe.loc[index, 'file_name'].tolist()
                    # moments = lframe[['moment' + str(i) for i in range(6)]].to_numpy()
                    moments = t.Tensor(lframe.loc[index,
                        ['moment' + str(i) for i in range(6)]].values)/\
                              t.Tensor([
                            5.04089976e+08, 1.64563652e+08, 5.75078730e+08,
                            6.47246460e+07, 2.00929929e+08, 7.24817978e+08])
                    moments = moments.cuda() if iscuda else moments
                    # angles_list = t2int(lframe.loc[idx, 'zero_angle'].tolist())
                    # angles_list = sample_batched['angles']
                    sz = inputs.shape
                    cnum = sz[1]
                    if all([lb == 'pc' or lb == 'f', opt.gttype == 'single_f_n',
                        opt.merging_order != 'color_channel']):
                        pathes = [j for j in pathes for i in range(enim)]
                        angles_list = t.flatten(angles_list)
                    if all([lb == 'pc' or lb == 'f', opt.gttype == 'single_f_n']):
                        # print(pathl)
                        # pathl = [path.replace(opt.csvname+'/','').replace('/','_')\
                        # for path in pathes]
                        # # print(pathl)
                        # indices = []
                        # for path in pathl:
                        #     indices.append([i for i, s in\
                        #      enumerate(F_N_names) if path in s][0])
                        # # print(indices)
                        # f_n = F_Nw[indices,:]
                        # # print('pathl=,',pathl)
                        # indices = []
                        # for path in pathl:
                        #     indices.append([i for i, s in\
                        #      enumerate(prmat_names) if path in s][0])
                        # print('indices,=', indices)
                        # prmat = bigm[index]
                        # print(index)
                        # print(prmat[0,0,:,:], prmat[1,0,:,:], prmat[2,0,:,:], prmat[3,0,:,:])
                        f_n = F_Nw[index,:]
                    if (opt.inputt == 'img' or opt.inputt == 'pc') and\
                            opt.merging_order == 'batch_channel':
                        inputs = t.Tensor(inputs).cuda() if iscuda else t.Tensor(inputs)
                        inputs = t.unsqueeze(t.reshape(inputs,
                            (sz[0]*sz[1], sz[2], sz[3])),axis = 1)
                    elif opt.inputt == 'img' or opt.inputt == 'pc' and\
                            opt.merging_order == 'color_channel':
                        # print(inputs.shape)
                        inputs = inputs.cuda() if iscuda else inputs
                        # inputs = t.Tensor(inputs).cuda() if iscuda else t.Tensor(inputs)

                    cbs = inputs.shape[0]
                    # print('inputs shape = ', inputs.shape)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    if opt.measure_time:
                        tf = time.time()
                    if opt.wandb and opt.measure_time:
                        wandb.log({'loading time '+phase: tf-ts})
                    if phase == 'train' and opt.measure_time:
                        lt[0, i_batch] = tf-ts
                    # forward
                    # track history if only in train
                    if lb != 'f_n' and not opt.rand_angle and\
                            opt.rot_dirs:
                        # print('make list of y_n2 for normal, last train and'\
                        #     'last validation batches')
                        y_n2 = t.zeros(cbs,nsp,opt.ampl)
                        y_n2 = y_n2.cuda() if iscuda else y_n2
                        for i, angle in enumerate(angles_list):
                            y_n2[i] = \
                                y_n[int(angle/10)*nsp:(int(angle/10) + 1)*nsp,:]
                    if lb != 'f_n' and opt.rot_dirs:
                        # print('make list of y_n2 for normal, last train and'\
                        #     'last validation batches')
                        dirs = t.zeros(cbs, nsp, 2)
                        dirs = dirs.cuda() if iscuda else dirs
                        for i, angle in enumerate(angles_list):
                          dirs[i] = \
                                bX[:, int(angle/10)*2:
                                    (int(angle/10) +
                                     1)*2]
                    if lb == 'pc' or lb == 'pc+f':
                        # making pc GT
                        GT = fn2p(y_n2[:cbs], f_n, dirs[:cbs], nsp, vox2mm, iscuda)

                        # np.save('C:/cherepashkin1/598test/plot_output/gt.npy', gt)
                        # print('GT shape after fn2p', GT.shape)
                        # gt = GT.detach().cpu().numpy()
                        # np.save('C:/cherepashkin1/598test/plot_output/gt.npy', GT.detach().cpu().numpy())
                        # print('GT stats 1', getstat(GT[0,0]))
                        cms = t.Tensor([lframe.loc[index, 'x_cms'].tolist(),
                                  lframe.loc[index, 'y_cms'].tolist(),
                                  lframe.loc[index, 'z_cms'].tolist()])
                        cms = cms.cuda() if iscuda else cms
                        # TODO check whether it makes sense to shift cms and then subtract it again, if matrix only translate
                        # GT = GT+cms.T.repeat(nsp,1,1).permute(1, 2, 0)
                        # print('GT stats 2', getstat(GT[0,0]))
                        GT = t.cat((GT, t.ones(nsp).repeat(GT.shape[0], 1, 1).cuda()), axis=1)
                        GT = t.einsum('nji,nkj->nik', GT, matw[index, :, :])
                        GT = (GT / t.unsqueeze(GT[:, :, 3], axis=2))[:, :, :3]
                        GT = t.transpose(GT, 1, 2)

                        # GT = t.cat(
                        #     (GT.reshape([GT.shape[0], 3, -1]),
                        #      t.ones(nsp).repeat(GT.shape[0], 1, 1).cuda()),
                        #     axis=1)
                        # # print(prmat.shape)
                        # # mat = t.einsum('ij, kjn-> kin',
                        # #                basisminv, prmat[:, 0, :, :])
                        # # print(mat)
                        # # GT = t.einsum('ijk,ikn->ijn', GT, mat)
                        # np.save('C:/cherepashkin1/598test/plot_output/gt2.npy',
                        #         GT.detach().cpu().numpy())
                        # print('index', index)
                        # GT = t.einsum('ijk,ijn->ikn', GT, matw[index,:,:])
                        # # print(GT.shape)
                        # GT = (GT / t.unsqueeze(GT[:, :, 3], axis=2))[:, :, :3]
                        # GT = t.transpose(GT, 1, 2)
                        # # print('GT shape after shift', GT.shape)
                        # print(GT.shape)
                        # print('GT stats after shift', getstat(GT[1,0,:]))
                        # sys.exit()
                        # print('GT shape', GT.shape)
                        # GT = GT-cms.T.repeat(nsp,1,1).permute(1, 2, 0)
                        # print('GT stats 3', getstat(GT[0,0]))
                        # GT*=vox2mm
                        # print('GT.shape', GT.shape)
                        # print(t.mean(GT[0,:,:], axis=1))
                    if lb == 'pc' and opt.merging_order != 'color_channel':
                        for i in range(cbs):
                            p[i, :, :] = t.matmul(
                                t.transpose(t.squeeze(
                                    C[int(angles_list[i]/10), :, :]), 0, 1), p[i, :, :])
                    if lb == 'f':
                        # print(f_n.shape, y_n2.shape)
                        GT = t.einsum('bh,bph->bp', f_n, y_n2[:cbs])
                    with t.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        if opt.measure_time:
                            ts = time.time()
                        if (opt.inputt == 'img' or\
                                opt.inputt == 'f') and\
                                not(lb == 'pc+f' or lb == 'pose6')\
                                and not opt.outputt == 'f_n':
                            outputs, latent = model(inputs)
                            # print('outputs', outputs.shape)
                        elif (opt.inputt == 'img' or\
                                opt.inputt == 'f') and\
                                not lb == 'pc+f' and\
                                (opt.outputt == 'f_n' or opt.outputt == 'pose6'):
                            outputs = model(inputs)
                        # print('outputs.shape', outputs.shape)
                        if opt.measure_time:
                            tf = time.time()
                        if opt.wandb and opt.measure_time:
                            wandb.log({'model output time '+phase: tf-ts})
                        if phase == 'train' and opt.measure_time:
                            lt[1, i_batch] = tf-ts
                        if opt.measure_time:
                            ts = time.time()

                        if mt:
                            print('calculate loss', time.time())
                        if opt.lb == 'pc' and opt.rotate_output and\
                            opt.outputt != 'f_n':
                            outputs_1 = t.reshape(outputs,
                            (outputs.shape[0],3,nsp))
                            outputs_1 = outputs_1.cuda() if iscuda else outputs_1
                            outputs_2 = t.zeros(cbs,3,nsp)
                        elif all([opt.lb == 'pc' or opt.lb == 'f',
                            opt.outputt == 'f_n',
                                opt.merging_order == 'color_channel']):
                            # print('outputs.shape, y_n2.shape', outputs.shape, y_n2.shape)
                            outputs_2 = t.einsum('bh,bph->bp', outputs, y_n2[:cbs])
                        elif all([opt.lb == 'pc' or opt.lb == 'f',
                            opt.outputt == 'pc',
                                opt.merging_order == 'color_channel']):
                            # print('outputs.shape, y_n2.shape', outputs.shape, y_n2.shape)
                            outputs_2 = outputs.reshape([outputs.shape[0], 3, -1])
                        if opt.lb == 'pc' and opt.rotate_output and\
                            opt.merging_order != 'color_channel' and\
                             opt.outputt != 'f_n':
                            for i in range(cbs):
                                outputs_2[i,:,:] = t.matmul(
                                    t.transpose(t.squeeze(
                                        C[int(angles_list[i]/10),:,:]),0,1),
                                        outputs_1[i,:,:])
                        if opt.lb == 'pc' and opt.rotate_output:
                            outputs_2 = outputs_2.cuda() if iscuda else outputs_2
                        # if all([lb == 'f', opt.criterion == 'L1',
                        #         opt.outputt != 'f_n']):
                        #     loss = vox2mm*t.mean(
                        #         t.abs(GT-\
                        #             opt.pscale*t.norm(
                        #             outputs_2.reshape([outputs_2.shape[0], 3, -1]),
                        #             dim=1)))
                        if all([lb == 'pc', opt.criterion == 'L1',
                                opt.outputt != 'f_n', not opt.rotate_output,
                                opt.merging_order == 'color_channel']):
                            loss = vox2mm * t.mean(
                                t.abs(GT -opt.pscale * outputs_2))
                        if all([lb == 'pc', opt.criterion == 'L1n',
                                opt.outputt != 'f_n', not opt.rotate_output,
                                opt.merging_order == 'color_channel']):
                            # print('before loss norm', t.abs(GT -opt.pscale * outputs_2).shape)
                            # sys.exit()
                            loss = vox2mm * t.mean(
                                t.norm(GT -opt.pscale * outputs_2, dim=1))
                        if all([lb == 'pose6', opt.outputt == 'pose6',
                                opt.inputt == 'img']):
                            # print('890', outputs.shape, moments.shape)
                            loss = t.sqrt(t.sum((outputs - moments)**2)/6)
                        # if all([lb == 'pc' or lb == 'f', opt.criterion == 'L1',
                        #         opt.outputt == 'f_n']):
                        #     loss = vox2mm*t.mean(
                        #         t.abs(GT-outputs_2))
                        # if all([lb == 'pc' or lb == 'f', opt.criterion == 'L2',
                        #         opt.outputt == 'f_n']):
                        #     loss = vox2mm*t.sqrt(t.sum(
                        #         (GT-outputs_2)**2)/\
                        #            (GT.shape[0]*GT.shape[1]*GT.shape[2]))
                        if opt.measure_time:
                            tf = time.time()
                        if opt.wandb and opt.measure_time:
                            wandb.log({'loss calc time '+phase: tf-ts})
                        if phase == 'train' and opt.measure_time:
                            lt[2, i_batch] = tf-ts
                        if mt:
                            print('make backward pass', time.time())
                        # tbn = len(dataloaders['train'])
                        # print(loss.item())
                        if phase == 'train' and i_batch%\
                            (int(1/opt.updateFraction))==0:
                            if opt.measure_time:
                                ts = time.time()
                            loss.backward()
                            optimizer.step()

                            if opt.measure_time:
                                tf = time.time()
                                lt[3, i_batch] = tf-ts
                        # print('outputs_2 stat 3', getstat(opt.pscale*outputs_2[1,2,:]))

                    # np.savetxt(jn(dirname,'outputs_2_'+phase+'_'+\
                    #     str(epoch).zfill(3)+'_'+\
                    #     str(i_batch).zfill(3)),
                    #     outputs_2[0].cpu().detach().numpy(),delimiter=',')
                    if lb == 'pc+f':
                        rloss0 += loss0.item()
                        rloss1 += loss1.item()
                    else:
                        rloss += loss.item()
                    # opt.machine != 'jureca'
                    cond = all([rank == 0, opt.save_output,
                        i_batch % opt.batch_output == 0, i_batch > 0,
                         lb == 'pc' or lb == 'f', epoch > epoch0])
                    if rank == 0 and i_batch % 10 == 0:
                        print('batch {}, {} loss = {:.2f}, '
                              'mean loss = {:.2f}'.format(i_batch, phase,
                                                          loss.item(),
                                                          rloss/(i_batch+1)))
                        # lossarb.append(rloss)
                        print(time.ctime())

                    if cond and not opt.minmax:
                        gtb = GT.detach().cpu().numpy()
                        # print('np.mean(gt)=',np.mean(gtb))
                        gtb0 = gtb[0]
                        # gt[pcnt] = np.reshape(gt[pcnt],(gt[pcnt].shape[0],-1))
                        # outputs_2 = t.reshape(outputs_2, (cbs,-1))
                    if cond and not opt.minmax and opt.outputt != 'f_n':
                        ob = outputs_2.detach().cpu().numpy()*opt.pscale*vox2mm
                    elif cond and not opt.minmax and opt.outputt == 'f_n':
                        ob = fn2p(y_n2[:cbs], outputs, dirs[:cbs],
                            nsp, vox2mm, iscuda)
                    if cond and epoch == epoch0+1:
                        Path(jn(dirname,'perbatch_showpoint')).\
                            mkdir(parents=True, exist_ok=True)
                        np.savetxt(jn(dirname,'perbatch_showpoint',\
                         'gtb_'+phase+\
                         '_' + str(i_batch).zfill(3)), gtb.reshape(cbs, -1))
                    if cond and opt.lb != 'f':
                        oob = ob[0]
                        oob = oob.reshape((3, nsp))
                    if cond and opt.lb == 'f' and opt.outputt == 'f_n':
                        oob = ob[0]
                        # oob = oob.reshape((3, nsp))
                        showmanypoints(cbs,nim,ob,gtb,pathes,angles_list,phase,
                            i_batch,cnt,jn(dirname,'showPoints','perbatch_showpoint'),
                            opt.merging_order, vox2mm)
                        np.savetxt(jn(dirname,'perbatch_showpoint',\
                         'o_'+phase+'_' + str(cnt).zfill(3) +\
                         '_' + str(i_batch).zfill(3)), ob.reshape(cbs, -1))
                        print('curloss for %s phase for %d epoch for %d batch = %f'\
                            %(phase,epoch,i_batch,np.mean(
                            np.abs(LA.norm(oob,axis=0)-LA.norm(gtb0,axis=0)))))
                        print(time.ctime())
                if opt.wandb and opt.measure_time:
                    wandb.log({'backward time '+phase: tf-ts})
                if lb == 'pc+f':
                    lossar[pcnt][epoch] = rloss0/(i_batch+1)
                    lossar[pcnt+2][epoch] = rloss1/(i_batch+1)
                else:
                    lossar[pcnt][epoch] = rloss/(i_batch+1)
                if opt.wandb:
                    wandb.log({phase+' loss': lossar[pcnt][epoch]})
                st1 = time.time()
                if mt:
                    print('save preliminary output',  time.time())
                log1 = all([rank == 0, epoch > epoch0, opt.save_output,
                    lb != 'f_n'])
                if epoch == epoch0+1 and opt.save_output and rank == 0 and\
                    phase == 'train':
                    original_stdout = sys.stdout
                    with open(jn(dirname,"opt.txt"), 'a') as f:
                        sys.stdout = f  # Change the standard output to the file we created.
                        print(opt)
                    with open(jn(dirname,"sys_argv.txt"), 'a') as f:
                        sys.stdout = f  # Change the standard output to the file we created.
                        print(sys.argv)
                    sys.stdout = original_stdout
                    # print('opt:\n',opt)
                    # print('sys.argv:\n',sys.argv)
                    # with open(jn(dirname,"job-parameters.txt"), "w") as f:
                    #     f.write(parameters_list)
                # if log1 and opt.minmax and not opt.outputt == 'f_n':
                #     ampl = opt.ampl
                #     o = np.multiply(
                #         outputs_2.detach().cpu().numpy(),
                #         tmean[3, :ampl] -
                #         tmean[2, :ampl]) + tmean[2, :ampl]
                #     gt = np.multiply(
                #         f_n.detach().cpu().numpy(),
                #         tmean[3, :ampl] -
                #         tmean[2, :ampl]) + tmean[2, :ampl]
                if all([log1, not opt.minmax,
                        not (opt.outputt == 'f_n' or opt.outputt == 'pose6')]):
                    o = outputs_2.detach().cpu().numpy()
                    # print('o stat1', getstat(o[0,2,:]))
                    # print('o shape = ', o.shape)
                elif log1 and not opt.minmax and opt.outputt == 'f_n':
                    o = (fn2p(y_n2[:cbs], outputs, dirs[:cbs],
                        nsp, vox2mm, iscuda)).detach().cpu().numpy()
                # print('o shape', o.shape)
                if all([rank == 0, opt.save_output, lb == 'pc' or lb == 'f', not opt.minmax,
                epoch > epoch0]):
                    o = o*opt.pscale*vox2mm
                    # print('o before showmanypoints', getstat(o[0,2,:]))
                if opt.merging_order == 'color_channel':
                    i0 = np.squeeze(inputs[0,0].detach().cpu().numpy())
                else:
                    i0 = np.squeeze(inputs[0].detach().cpu().numpy())
                # if all([rank == 0, opt.save_output, lb != 'f_n', not opt.minmax,
                #         epoch == epoch0+1, opt.merging_order != 'color_channel',
                #         opt.outputt != 'f_n']):
                #     gt.append(GT.detach().cpu().numpy())
                #     # print(gt[0].shape)
                #     gt0.append(gt[pcnt][0])
                #     # gt[pcnt] = np.reshape(gt[pcnt],(gt[pcnt].shape[0],-1))
                #     curangle = int(angles_list[0]/10)
                #     backproject(prmat[4*curangle:4*(curangle+1)],i0, gt0[pcnt],
                #         jn(dirname,phase))
                if all([rank == 0, opt.save_output, not(lb == 'f_n' or lb == 'pose6'), not opt.minmax,
                        epoch == epoch0+1, opt.merging_order == 'color_channel',
                        opt.outputt == 'f_n' or opt.outputt == 'pc']):
                    # print(GT.shape)

                    # gt.append(f2p(GT, dirs[:cbs], nsp, vox2mm).detach().cpu().numpy())
                    gt.append(GT.detach().cpu().numpy())
                    # gt.append(GT.reshape(GT.shape[0],-1))
                    gt0.append(gt[pcnt][0])
                    curangle = int(angles_list[0][0]/10)
                    # print('agnles shape', angles_list.shape)
                    # print('prmat shape', prmat.shape)

                    # print('gt stat = ', np.min(gt0[pcnt]), np.mean(gt0[pcnt]), np.max(gt0[pcnt]))
                    # print('gt0[pcnt].shape',gt0[pcnt].reshape(3,nsp).shape)
                    # print('gt stat', getstat(gt0[pcnt]))
                    # print('curangle', curangle)
                    # np.savetxt('C:/cherepashkin1/gt0', gt0[pcnt], delimiter=',')
                    # np.save('C:/cherepashkin1/prmat',prmat.detach().cpu().numpy())
                    # np.savetxt('C:/cherepashkin1/i0', i0, delimiter=',')
                    # print('before backproject', prmatw[index[0], curangle, :, :].shape, i0.shape, gt0[pcnt].shape)
                    backproject(prmatw[index[0], curangle, :, :].detach().cpu().numpy(),i0,
                        gt0[pcnt].reshape(3,nsp),
                        jn(dirname,phase), opt.rotate_output)
                    # sys.exit()
                # elif all([rank == 0, opt.save_output, lb != 'f_n', not opt.minmax,
                #         epoch == epoch0+1, opt.merging_order == 'color_channel',
                #         opt.outputt != 'f_n', nim>1]):
                #     gt.append(GT.detach().cpu().numpy())
                #     print('gt[0].shape', gt[0].shape)
                #     gt0.append(gt[pcnt][0])
                #     # gt[pcnt] = np.reshape(gt[pcnt],(gt[pcnt].shape[0],-1))
                #     curangle = int(angles_list[0][0]/10)
                #     backproject(prmat[4*curangle:4*(curangle+1)],i0,
                #         np.squeeze(f2p(np.expand_dims(gt0[pcnt],axis=0),
                #          dirs[0], nsp, vox2mm)),
                #         jn(dirname,phase))
                if log1 and phase == 'val' and opt.measure_time:
                    fig = plt.figure()
                    for i in range(4):
                        plt.plot(lt[i,:])
                        # plt.xlim(0,epoch*bs)
                    plt.savefig(dirname+'lt.png')
                    plt.close(fig)
                    np.savetxt(dirname+'lt',lt)
                if all([rank == 0, epoch == epoch0+1, opt.save_output,
                    not(lb == 'f_n' or lb == 'pose6')]):
                    savelist(pathes, jn(dirname,"pathes_"+phase+".txt"))
                    try:
                        savelist([str(int(i/10)) for i in angles_list.tolist()],
                            jn(dirname,"angles_"+phase+".txt"))
                    except:
                        savelist([str(int(i/10)) for i in\
                            t.flatten(angles_list).tolist()],
                            jn(dirname,"angles_"+phase+".txt"))
                if all([log1, not (lb == 'f_n' or lb == 'pose6'), epoch == epoch0+1]):
                    # try:
                    #     print(gt.shape)
                    # except:
                    #     print(gt[0].shape)
                    Path(jn(dirname,'netOutputs')).mkdir(parents=True, exist_ok=True)
                    np.savetxt(jn(dirname,'netOutputs','gt_'+phase),
                               np.reshape(gt[pcnt],(cbs,-1)), delimiter=',')
                    # print('inputs[0].shape=',inputs[0].shape)
                    np.savetxt(jn(dirname,'input_image_'+phase),i0,delimiter=',')
                if rank == 0 and epoch == epoch0+1 and opt.save_output:
                    for n in opt.netname:
                        shutil.copy(jn(homepath, 'circles/finetune_test/experiments',
                                                 opt.expnum, n+'.py'),
                                                    jn(dirname,n+'.py'))
                    shutil.copy(jn(homepath, finePath, 'main.py'),
                                jn(dirname, 'main.py'))
                    shutil.copy(jn(finePath, "transform"+opt.transappendix+".py"),
                                jn(dirname, "transform"+opt.transappendix+".py"))
                if all([rank == 0, lb == 'pc' or lb == 'f', opt.save_output, epoch > epoch0]):
                    oo = o[0]
                    # print('oo.shape',oo.shape)
                    oo = oo.reshape((3, nsp))
                # elif all([rank == 0, lb == 'f', opt.outputt == 'f_n', opt.save_output, epoch > epoch0]):
                #     oo = o[0]
                if all([rank == 0, not(lb == 'f_n' or lb == 'pose6'), opt.save_output, epoch > epoch0]):
                    # print('gt shape', gt[pcnt].shape)
                    # print('o shape', o.reshape((-1,3,nsp)).shape, gt[pcnt].shape)
                    # print('o stat', getstat(o.reshape((-1,3,nsp))[0,2,:]), getstat(vox2mm*gt[pcnt][0,2,:]))
                    showmanypoints(cbs,nim,o.reshape((-1,3,nsp)),vox2mm*gt[pcnt],
                        pathes,angles_list,phase,i_batch,cnt,jn(dirname, 'showPoints'),
                        opt.merging_order, vox2mm)
                    curloss[pcnt,epoch] = np.mean(np.abs(LA.norm(oo,axis=0)-\
                        LA.norm(gt0[pcnt],axis=0)))

                    print('curloss for %s phase for %d epoch = %f'\
                        %(phase,epoch,curloss[pcnt,epoch]))
                if rank == 0 and lb == 'pc' and\
                    opt.save_output and epoch > epoch0 and opt.wandb:
                        wandb.log({phase+"_points": wandb.Image(jn(dirname,'pc_'+\
                        phase+'_' +str(cnt).zfill(3) + '.png'))})
                if rank == 0 and not(lb == 'f_n' or lb == 'pose6') and\
                    opt.save_output and epoch > epoch0 and opt.wandb:
                    wandb.log({"point_cloud " + phase:\
                        wandb.Object3D(oo.transpose())})
                if rank == 0 and not(lb == 'f_n' or lb == 'pose6') and\
                    opt.save_output and epoch > epoch0 and opt.wandb and\
                     phase == 'val':
                        wandb.log({'train curloss': curloss[0,epoch],\
                         'val curloss': curloss[1,epoch]})
                if log1 and phase == 'val':
                    lossfig(jn(dirname,'learning_curve_'), lossar,
                     'Loss', 'Learning curve', (0,epoch), (0,1), lb)
                    logloss = np.ma.log10(lossar)
                    lossfig(jn(dirname,'learning_curve_'), logloss.filled(0),
                        'log10(Loss)', 'Learning curve', (0,epoch), (0,0), lb)
                    lossfig(jn(dirname,'curloss_'), curloss,
                     'Loss', 'Loss for single seed', (0,epoch), (0,1), lb)
                    # lossfig(dirname+'curloss_', np.abs(curloss),
                    #  'Abs Loss', 'Loss for single seed', (0,epoch), (0,0.2), lb)
                    logloss = np.ma.log10(np.abs(curloss))
                    lossfig(jn(dirname,'curloss_'), logloss.filled(0),
                        'log10(abs loss)', 'Loss for single seed', (0,epoch),
                         (0,0), lb)
        #                 lossars = np.array([np.trim_zeros(lossar[0,:]),
    #                     np.trim_zeros(lossar[1,:]), ])

    #                 print(lossars)
                if (rank == 0 and
                        epoch > epoch0 and
                        opt.save_output and
                        not(lb == 'f_n' or lb == 'pose6')):
                    # print(o.shape)
                    # model.eval()
                    # with t.no_grad():
                    #     o3,latent3 = model(t.unsqueeze(inputs[0],axis=0))
                    #     o4,latent4 = model(inputs)
                    #     # t.save(model.state_dict(), jn(dirname,"model2_"+\
                    #     #     str(cnt).zfill(3)))
                    #     np.savetxt(jn(dirname,'o3_'+phase+'_' +\
                    #                str(cnt).zfill(3)), o3.cpu().detach().numpy(), delimiter=',')
                    #     np.savetxt(jn(dirname,'latent3_'+phase+'_' +\
                    #                str(cnt).zfill(3)), latent3.cpu().detach().numpy(), delimiter=',')
                    #     np.savetxt(jn(dirname,'o4_'+phase+'_' +\
                    #                str(cnt).zfill(3)), o4.cpu().detach().numpy(), delimiter=',')
                    #     np.savetxt(jn(dirname,'latent4_'+phase+'_' +\
                    #                str(cnt).zfill(3)), latent4.cpu().detach().numpy(), delimiter=',')
                    Path(jn(dirname,'netOutputs')).mkdir(parents=True, exist_ok=True)
                    np.savetxt(jn(dirname,'netOutputs','o_'+phase+'_' +\
                               str(cnt).zfill(3)), o.reshape(-1,3*nsp), delimiter=',')

                    # np.save(dirname + 'lossar.npy', lossar)
                    np.savetxt(jn(dirname,'lossar'), lossar, delimiter=',')
                    np.savetxt(jn(dirname,'curloss'), curloss, delimiter=',')
                if (rank == 0 and
                        epoch > 0 and
                        opt.save_output and
                        not(lb == 'f_n' or lb == 'pose6')):
                    t.save(model.state_dict(), jn(dirname,"model"))
                        # str(cnt).zfill(3))
                         # and epoch % opt.ufmodel == 0
                    if opt.wandb:
                        wandb.watch(model)
                if all([rank == 0, epoch > epoch0, opt.save_output,
                        not(lb == 'f_n' or lb == 'pose6')]):
                    Path(jn(dirname,'latent')).mkdir(parents=True, exist_ok=True)
                    np.savetxt(jn(dirname,'latent','latent_'+phase+'_' +\
                               str(cnt).zfill(3)), latent.detach().cpu().numpy(), delimiter=',')
                # GIF movie making
                # if rank == 0 and opt.save_output and\
                #     (lb == 'pc' or lb == 'pc+f') and epoch > epoch0:
                #     cip = []
                #     for root, directories, filenames in os.walk(dirname):
                #         for filename in filenames:
                #             if 'pc_'+phase in filename and not 'checkpoint' in filename:
                #                 cip.append(jn(root,filename))
                #     cip.sort()
                #     images = []
                #     for filename in cip:
                #         images.append(imageio.imread(filename))
                #     imageio.mimsave(jn(dirname, phase+'_movie.gif'), images, duration=0.5)
                losst = 0 if phase == 'train' else 1
                if rank == 0:
                    print('{} Loss: {:.2f}'.format(phase, lossar[pcnt][epoch]))
                    print()
                # print('time after epoch %s' %(time.time()-st1))
            cnt += 1
            scheduler.step()
            if rank == 0:
                print('epoch %d was done for %f seconds' %(epoch, time.time()-ste))
        time_elapsed = time.time() - since
        if rank == 0:
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        return model, lossar, time_elapsed

    #@simple_time_tracker(_log)
    class Seed3D_Dataset(Dataset):
        """seed point cloud dataset."""

        def __init__(self, path, lframe, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.df = lframe
            # print('length of self df =',len(self.df))
            self.root_dir = path
            self.transform = transform

        def __len__(self):
            return len(self.df)

        #@simple_time_tracker(_log)
        def __getitem__(self, idx):
            # idx = idx.tolist()
            # print('idx', idx, type(idx))
            # print(self.df.head())
            # print('idx inside dataset', idx)
            # fln = self.df._get_value(idx, 'file_name')
            fln = self.df.iloc[idx]['file_name']
            # index = self.df.iloc[idx]['index']
            # fln = self.df.loc[idx, 'file_name']
            #ts = time.time()
            if t.is_tensor(idx):
                pass
            img = 0
            # print('df len =', len(self.df))
            # fln = self.df.loc[idx, 'file_name']
            if opt.inputt == 'pc':
                img_name = jn(self.root_dir, self.root_dir.replace(
                    'phenoseed_csv', 'phenoseed'),
                              fln.replace('csv', '')+
                              '_Surface.ply').replace('\\', '/')
                # self.df._get_value(idx, 'file_name')
                pcd = np.asarray(open3d.io.read_point_cloud(img_name).points)
                # img = np.asarray(pcd.points)
                img = np.concatenate(
                    (img, np.zeros([58014 - img.shape[0], 3])), axis=0)
            elif opt.inputt == 'img':
                t.manual_seed(idx)
            if opt.inputt == 'img' and opt.noise_input:
                img = t.rand(enim, original_h, original_w)
                # angles_list = np.array([0])
            # print(inn and not(opt.rand_angle or opt.view_sep or opt.zero_angle),
            # inn and not(opt.rand_angle or opt.view_sep) and opt.zero_angle)
            # print('rand_angle', opt.rand_angle)
            # print('view_sep', opt.view_sep)
            # print('zero_angle', opt.zero_angle)
            # print('inn', inn)

            # if inn and not(opt.rand_angle or opt.view_sep or opt.zero_angle):
            #     angles_list = np.zeros(alf.shape)
            #     for i, angle in enumerate(alf):
            #         newa = angle - 10*int(self.df.iloc[idx, 1])
            #         angle = newa if newa >= 0 else newa+360
            #         angles_list[i] = int(angle)

            # elif inn and not(opt.rand_angle or opt.view_sep) and opt.zero_angle and\
            #         enim > 1:
            #     angles_list = np.zeros(alf.shape)
            #     for i, angle in enumerate(alf):
            #         newa = angle - 10*int(self.df.iloc[idx, 1])
            #         angle = newa if newa >= 0 else newa+360
            #         angles_list[i] = int(angle)
            if opt.inputt == 'img' and opt.noise_input:
                angles_list = np.array([0])
            if inn and not(opt.rand_angle or opt.view_sep) and opt.zero_angle:
                angles_list = np.array(alf)
            elif inn and not(opt.rand_angle or opt.zero_angle or opt.view_sep):
                angles_list = np.array([10*self.df.loc[idx,'zero_angle']])
            angles_list = angles_list.astype(int)
            # print(angles_list)

            #         for i in range(len(angles_list)):
            #             angles_list[i] = int(angles_list[i])
            # print(angles_list)
            # print(angles_list)
            if inn:
                # st = time.time()
                img_name = []
                img = np.zeros([enim, original_h, original_w]).astype(np.single)
                # print(angles_list)
                for i in range(enim):
                    img_name = jn(
                        self.root_dir.replace('phenoseed_csv', 'phenoseed'),
                        fln.replace(opt.csvname, opt.specie),
                        'rotation_' + str(angles_list[i]).zfill(3) +
                        '.tif').replace('\\', '/')
                    # print('img name', img_name)
                    #                 img[i] = np.expand_dims(np.asarray(io.imread(img_name)),
                    #                                                         axis=2)
                    # st = time.time()
                    curim = np.asarray(io.imread(img_name), dtype=np.single)
                    # print('curim dtype() = ', curim.dtype)

                    # print('one image loaded in %f seconds' %(time.time()-st))

                    h1, w1 = curim.shape
                    if (h1, w1) == (original_h, original_w):
                        img[i] = curim
                    else:
                        h2, w2 = original_h, original_w
                        th, tw = int((h2-h1) / 2), int((w2-w1) / 2)
                        img[i] = np.pad(curim, ((th, th), (tw, tw)))
                # img = 1 - img/255
                # print('one image loaded in %f seconds' %(time.time()-st))
                # img = img
                # print('3D img.dtype() = ', img.dtype)
            if not opt.noise_output and not opt.gttype == 'single_f_n':
                fara = np.genfromtxt(jn(self.root_dir, (
                        fln +
                        '_Far_' + str(nsp) + classicnorm +
                        '.csv')).replace('\\', '/'), delimiter=',')
                f_na = np.genfromtxt(jn(self.root_dir, (
                        fln +
                        '_F_N_' + str(nsp) + classicnorm +
                        '.csv')).replace('\\', '/'), delimiter=',')
            elif opt.noise_output:
                far = t.rand(nsp)
                f_n = t.rand(opt.ampl)
            if not opt.noise_output and not opt.rand_angle and \
                    enim == 1 and not opt.gttype == 'single_f_n':
                far = fara[int(angles_list[0]/10)]
                f_n = f_na[int(angles_list[0]/10)]
            # path = jn(self.df.iloc[idx, 0])
            if not opt.gttype == 'single_f_n':
                sample = {'image': img, 'idx': idx}
            elif opt.gttype == 'single_f_n':
                # f_n = np.genfromtxt(jn(self.root_dir, (
                #     self.df.iloc[idx, 0] +
                #     '_F_N.csv')).replace('\\', '/'), delimiter=',')
                # print('f_n.shape',f_n.shape)
                #             print('img:',type(img), ' ', img.shape)
                #             print('f_n:',type(f_n), ' ', f_n.shape)
                #             print('angles:',angles_list)
                #             print('path:',path)
                # path2 = [path]*enim
                sample = {'image': img}
            # print(angles_list)
            # print(sample['image'])
            # print('sample[image].dtype=',sample['image'].dtype)
            if self.transform:
                # print(self.transform)
                st = time.time()
                sample = self.transform(sample)
                # print('one transform was done in %f seconds' %(time.time()-st))
            #tf = time.time()
            # if opt.wandb:
            #     wandb.log({'loading single data sample ': tf-ts})
            # print(sample['angles'])
            # print(sample['path'])
            # print('#2 length of self df =',len(self.df))
            # print('sample[image].dtype=',sample['image'].dtype)
            # print('sample[image].dtype=',sample['image'].type())
            return sample, self.df.iloc[idx]['index'], angles_list
    #print(__name__)
    #exec(open(homepath+"circles/finetune_test/transform_f_n.py").read())
    # if not(opt.single_folder or opt.outputt == 'single_f_n'):
    #     exec(open(homepath+"circles/finetune_test/transform.py").read())
    # elif not opt.single_folder and opt.outputt == 'single_f_n':
    #     exec(open(homepath+"circles/finetune_test/transform_f_n.py").read())
    # elif opt.laptop:
    #     exec(open(
    #         path1,
    #         "transform_f_n.py")).read())
    # elif opt.single_folder and not opt.laptop and not opt.outputt == 'single_f_n':
    #     exec(open("transform.py").read())
    exec(open(jn(finePath, "transform"+opt.transappendix+".py")).read())
    # if opt.outputt == 'single_f_n':
    #     exec(open(jn(path1, "transform_f_n.py")).read())
    # else:
    #     exec(open(jn(path1, "transform.py")).read())



    if mt:
        print('define transforms options')
    batch_size = opt.bs
    if not opt.noise_output and opt.minmax:
        tmean = np.genfromtxt(
            jn(finePath,'csv','tmean.csv'),
            delimiter=',')
    else:
        tmean = np.zeros([5270, opt.ampl])
    minmax, minmax3dimage, normalize, center, cmscrop, cencrop, downsample = ['']*7
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
    if not opt.inputt == 'pc' and opt.cencrop:
        cencrop = 'CentralCrop(opt.cencrop),'
    else:
        cencrop = ''
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
    if opt.inputt == 'img':
        standardize = 'Standardize(opt.standardize), '
    else:
        standardize = ''
    if opt.noise_input or opt.single_folder:
        (minmax, minmax3dimage, normalize, center,
         cmscrop, cencrop, downsample, minmax_f) = ['']*8
        rescale = 'Rescale(opt.rescale), '
    es0 = "transforms.Compose([" +\
         cencrop+cmscrop+rescale+standardize+minmax3dimage+normalize+center+\
         downsample+"])"
    # es = ()
    # es = ("data_transforms = {\'train\': transforms.Compose([" +
    #      minmax + minmax3dimage + normalize + center +
    #      cmscrop + downsample + minmax_f + rescale +
    #      "AmpCrop(opt.ampl),ToTensor(iscuda)]),\
    #      'val': transforms.Compose([" +
    #      minmax + minmax3dimage + normalize + center +
    #      cmscrop + downsample + minmax_f + rescale +
    #      "AmpCrop(opt.ampl),ToTensor(iscuda)])}")
    # print(es)
    exec("data_transforms = {\'train\': "+es0+",'val': "+es0+"}")
    print(data_transforms)
    # print(data_transforms)
    # kwargs = {}
    # kwargs = {'num_workers': 1, 'pin_memory': True} if opt.parallel == 'hvd' else {}
    kwargs = {'num_workers': opt.num_workers, 'pin_memory': opt.pin_memory} if opt.parallel == 'hvd' else {}
    if (opt.parallel == 'hvd' and
        kwargs.get('num_workers', 0) > 0 and
        hasattr(mp, '_supports_context') and mp._supports_context and
            'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    #print("Initializing Datasets and Dataloaders...")

    if mt:
        print('loading of dataset csv with pathes to F_N and angles')
    excpath = jn(finePath,opt.specie+'_exceptions.txt')
    # mtl(mt1)
    csvPathSep = jn(homepath,'cherepashkin1', 'phenoseed_csv', opt.specie)
    if not os.path.isfile(excpath) and not opt.single_folder:
        cip = []
        # i = 0
        for root, directories, filenames in os.walk(csvPathSep):
            for filename in filenames:
                if 'F_N.csv' in filename:
        #             F_Nm[i,:] = np.genfromtxt(
        #                 jn(root,filename),delimiter=',')
        #             i+=1
                    cip.append(jn(root,filename))
        F_Nm = np.zeros([len(cip),opt.ampl])
        for i in range(len(cip)):
            F_Nm[i,:] = np.genfromtxt(cip[i],delimiter=',')
        for i in range(opt.ampl):
            ar = F_Nm[:,i]
            a = np.concatenate((a,np.where(np.abs(ar-np.mean(ar)) > 6*np.std(ar))[0]))
        exception_list = [cip[index][-23:-8] for index in np.unique(a.astype(int))]
        savelist(exception_list, excpath)
    elif os.path.isfile(excpath) and not opt.single_folder:
        with open(excpath, "r") as f:
            exception_list = f.readlines()
    else:
        exception_list = []
    print(len(exception_list))
    with open(excpath.replace('exceptions.txt', 'exceptions_good.txt'), "r") as f:
        good = f.readlines()
    # print(len(exception_list))
    # print(len(good))
    for g in good:
        exception_list = [ x for x in exception_list if g not in x ]
    # print(len(exception_list))
    # sys.exit()
    pts = jn(finePath, 'csv', opt.specie + 'frame.csv')
    print('pts',pts)
    #print(opt.noise_output, os.path.isfile(pts), opt.use_existing_csv)
    # def gf():
    #     exec("print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)")
    if mt1:
        frameinfo = getframeinfo(currentframe())
        print(frameinfo.filename, frameinfo.lineno)
    # print(opt.noise_output, os.path.isfile(pts), opt.use_existing_csv)
    # print(not opt.noise_output and os.path.isfile(pts) and\
    #     opt.use_existing_csv)
    if opt.noise_output:
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
        lframe = pd.DataFrame()
        lframe.insert(
            0, 'file_name', np.zeros(5270).astype(str))
    elif not opt.noise_output and os.path.isfile(pts) and\
            opt.use_existing_csv:
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
    #    print(1)
        # TODO use only one file for all F_Nw, prmat and zero angle
        lframe = pd.read_csv(pts)
        print("lframe's length after laoding = ", len(lframe))
    elif not opt.noise_output and not(os.path.isfile(pts) and\
            opt.use_existing_csv):
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
        cip = []
        for word in opt.specie.split(','):
            csvpath = jn(csvPathSep, word+'csv')
            for root, directories, filenames in os.walk(csvpath):
                for filename in filenames:
                    search_str = 'F_N_' + str(nsp) + '.csv'
                    for i in range(len(exception_list)):
                        if not any(exception in cip[i] for exception in exception_list):
    #                 if search_str in filename and flag:
                            tfolds = jn(root, filename).split('/')[-3:]
                            cip.append(jn(tfolds[0], tfolds[1], tfolds[2]))
        lframe = pd.DataFrame()
        lframe.insert(
            0, 'file_name', [cip[i].split('_F_N')[0] for i in range(len(cip))])

        list_zero = []
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)

        for idx in range(len(lframe)):
            s = np.zeros(36)
            rotpath = jn(
                dataPath,
                lframe.iloc[idx, 0].replace('csv', ''))
            for j in range(36):
                img = io.imread(
                    jn(
                        rotpath, 'rotation_' + str(10*j).zfill(3) + '.tif'))
                s[j] = np.sum(255-img)
            list_zero.append(np.argmax(s))
        lframe.insert(1, 'zero_angle', list_zero)
        lframe = lframe.sample(frac=1)
        lframe.to_csv(pts, index=False)
    inn = opt.inputt == 'img' and not opt.noise_input

    if inn and not opt.rand_angle and enim > 1:
        alf = np.array([10*int(36*i/nim) for i in range(nim)])
    elif inn and opt.rand_angle:
        alf = np.random.choice(
            np.arange(0, 360, 10), size=nim, replace=False)
    elif inn and nim == 1 and opt.zero_angle:
        alf = [0]
    # print(alf)
    pts = jn(finePath, 'csv', opt.specie + '_view_sep_'+str(nim)+'.csv')
    if opt.view_sep and os.path.isfile(pts) and\
             opt.use_sep_csv:
        lframe_sep = pd.read_csv(pts)
    else:
        lframe_sep = lframe

    # print(opt.view_sep, os.path.isfile(pts), opt.use_sep_csv,
    #         int(len(lframe_sep)/5270) != nim)
    isusable = os.path.isfile(pts) and\
             opt.use_sep_csv and int(len(lframe_sep)/5283) == nim
    if opt.view_sep and not(isusable):
        lframe_sep = pd.DataFrame(columns=('file_name', 'angle'))
        for idx in range(len(lframe)):
            # if mt1:
            #     frameinfo = getframeinfo(currentframe())
            #     print(idx, frameinfo.filename, frameinfo.lineno)
            for j, angle in enumerate(alf):
                if lframe.iloc[idx, 1]+int(angle/10) >= 36:
                    angle = lframe.iloc[idx, 1]+int(angle/10)-36
                else:
                    angle = lframe.iloc[idx, 1]+int(angle/10)
                lframe_sep.loc[idx*len(alf)+j] = [lframe.iloc[idx, 0], angle]
        # if mt1:
        #     frameinfo = getframeinfo(currentframe())
        #     print(frameinfo.filename, frameinfo.lineno)
        lframe = lframe_sep
        lframe.to_csv(pts, index=False)
    elif opt.view_sep and isusable:
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
        lframe = pd.read_csv(pts)

    # print(exception_list)
    for st in exception_list:
        # print(st)
        lframe = lframe[~lframe.file_name.str.contains(st.replace('\n', ''))]
    print('lframe len after excluding all exceptions=',len(lframe))
    # sys.exit()
    if opt.single_folder:
        lframe = lframe[:100]
    lframe = lframe.sort_values(by=['file_name'])
    lframe = lframe.sample(frac=1, random_state=0)
    lframe = lframe[:opt.framelim]
    # lframe = lframe[:min(len(lframe),opt.bn*opt.bs)]
    lframe = lframe.replace({opt.specie+'csv':opt.csvname}, regex=True)
    # print('lframe len=', len(lframe))
    train_part = int(0.8*len(lframe))
    df_dict = {'train': lframe[:train_part], 'val': lframe[train_part:]}
    print('len train = ', len(lframe[:train_part]))
    # Create training and validation datasets
    image_datasets = {x: Seed3D_Dataset(
        jn(homepath, 'cherepashkin1', 'phenoseed'), df_dict[x],
        transform=data_transforms[x]) for x in ['train', 'val']}

    if mt:
        print('define dataloaders and samplers')
    # if opt.parallel == 'hvd':
    #     samplers = {x: t.utils.data.distributed.DistributedSampler(
    #         image_datasets[x], num_replicas=hvd.size(),
    #         rank=hvd.rank(), shuffle=False) for x in ['train', 'val']}
    #     # Create training and validation dataloaders
    #     dataloaders_dict = {x: t.utils.data.DataLoader(
    #         image_datasets[x],
    #         batch_size=opt.bs, shuffle=False, sampler=samplers[x],
    #         worker_init_fn=seed_worker,
    #         generator=g, **kwargs) for x in ['train', 'val']}
    # elif opt.parallel == 'torch':
    #     samplers = None
    #     dataloaders_dict = {x: t.utils.data.DataLoader(
    #         image_datasets[x],
    #         batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)
    #         for x in ['train', 'val']}


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
    # bn = opt.bn
    opt.epoch = opt.epoch
    lossar = np.zeros([2, opt.epoch])
    # Train and evaluate
    # hidden_dim = np.array([int(i) for i in opt.hidden_dim])
    hidden_dim = np.array(opt.hidden_dim)
    # print(hidden_dim)
    chidden_dim = np.array(opt.chidden_dim)
    kernel_sizes = np.array(opt.kernel_sizes)
    # exec(opt.hidden_dim)
    # print(int(opt.chidden_dim.split(';')))
    # exec(opt.chidden_dim)
    # exec(opt.kernel_sizes)

    cropf = opt.cencrop if opt.cencrop else original_h
    rescalef = opt.rescale if cropf > opt.rescale else cropf

    def dpwrap(model):
        return(nn.DataParallel(model) if opt.parallel == 't' else model)
    if mt:
        print('load network architecture')
    if bool(opt.model_name):
        smodel, input_size = initialize_model(
            opt.model_name, opt.ampl, opt.feature_extract,
            use_pretrained=opt.use_pretrained)
    elif not bool(opt.model_name) and opt.inputt == 'img' and\
        opt.merging_order != 'latent_merge':
        # print('rescale', rescalef,
        #     int(rescalef*original_w/original_h))
        smodel = CNet(
            hidden_dim, chidden_dim, kernel_sizes,
            opt.num_input_images, rescalef,
            int(rescalef*original_w/original_h), opt.haf)
    elif not bool(opt.model_name) and opt.inputt == 'f' and\
        opt.merging_order != 'latent_merge':
        smodel = CNet(
            hidden_dim, chidden_dim, kernel_sizes,
            opt.num_input_images, 1, nsp, opt.haf)
    elif not bool(opt.model_name) and opt.inputt == 'pc' and\
        opt.merging_order != 'latent_merge':
        smodel = CNet(
            hidden_dim, chidden_dim, kernel_sizes,
            opt.num_input_images, np.floor(58014/opt.downsample).astype(int),
            1, opt.haf)
    elif opt.inputt == 'img' and opt.merging_order == 'latent_merge':
        smodel0 = Encoder(hidden_dim, chidden_dim, kernel_sizes,
                    opt.num_input_images, rescalef,
                    int(rescalef*original_w/original_h), opt.haf)
        smodel1 = Decoder(opt.num_input_images)
        smodel0 = dpwrap(smodel0)
        smodel1 = dpwrap(smodel1)
    if opt.conTrain:
        smodel.load_state_dict(t.load(jn(dir1,opt.conTrain,'model')))

    if opt.merging_order != 'latent_merge':
        smodel = dpwrap(smodel)
    lr = opt.lr*hvd.local_size() if opt.parallel == 'hvd' else opt.lr
    if iscuda and opt.merging_order != 'latent_merge':
        smodel.cuda()
    elif iscuda and opt.merging_order == 'latent_merge':
        smodel0.cuda()
        smodel1.cuda()
    optimizer = t.optim.Adam(
        smodel.parameters(), lr, betas=(0.9, 0.999), eps=1e-08,
        weight_decay=opt.weight_decay, amsgrad=False)
    # print(opt.steplr[0],type(opt.steplr[0]),opt.steplr[1],type(opt.steplr[1]))
    scheduler = StepLR(optimizer, step_size=opt.steplr[0], gamma=opt.steplr[1])
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
    # for pcnt, phase in enumerate(['train','val']):
    #     for i_batch, sample_batched in enumerate(dataloaders[phase]):
    #         img = sample_batched['img']
    #         bsa[pcnt, i_batch] = img.shape[0]
    # print(bsa)
    if mt:
        print('model .. = train_model(..)')
    # print('tttrain consists of %d images' %len(dataloaders_dict['train']))
    # model, lossar, time_elapsed = train_model(
    #     smodel, criterion, optimizer, tmean,
    #     opt.epoch=opt.epoch,
    #     is_inception=(opt.model_name == "inception"))
    model, lossar, time_elapsed = train_model(smodel, optimizer)

    if mt:
        print('saving output after training')
    if rank == 0 and opt.save_output:
        with open(jn(dirname,"job-parameters.txt"), 'a') as f:
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
    #             y = t.zeros([x.shape[0], x.shape[1]])
    #             y = y.cuda()
    #             for i in range(x.shape[0]):
    #                 y[i, :] = t.sqrt(
    #                     x[i, :, 0]**2 + x[i, :, 1]**2 + x[i, :, 2]**2)
    #             y = t.unsqueeze(y, 2)
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

        # t.save(model.state_dict(), dirname+"model")
        # for x in ['train', 'val']:
        #     df_dict[x].to_csv(dirname+'pathes_df_'+x+'.csv')
        print('ellapsed time = ', time.time()-tstart)
        # ## Return SH coefficient vector from the trained model
