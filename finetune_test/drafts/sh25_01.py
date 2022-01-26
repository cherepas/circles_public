#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
import argparse
import open3d
import torch
import horovod.torch as hvd
import torch.nn as nn
# import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
import torchvision
#from torchvision import datasets, models
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
#import copy
from skimage import io, transform
from skimage.measure import regionprops
from skimage import filters
from torch.utils.data import Dataset
import pandas as pd
from scipy.special import sph_harm
import sys
#import shutil
from numpy import linalg as LA
import datetime
#import gc
from standard_models import *

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('-bs', type=int, default=4)
parser.add_argument('-epoch', type=int, default=8)
parser.add_argument('-bn', type=int, default=8)
parser.add_argument('-lr', type=float, default=5e-5)
parser.add_argument('-minmax', dest='minmax', action='store_true')
parser.add_argument('-no_minmax', dest='minmax', action='store_false')
parser.set_defaults(minmax=False)
parser.add_argument(
    '-minmax3dimage', dest='minmax3dimage', action='store_true')
parser.add_argument(
    '-no_minmax3dimage', dest='minmax3dimage', action='store_false')
parser.set_defaults(minmax3dimage=False)
parser.add_argument('-normalize', dest='normalize', action='store_true')
parser.add_argument('-no_normalize', dest='normalize', action='store_false')
parser.set_defaults(normalize=False)
parser.add_argument('-center', dest='center', action='store_true')
parser.add_argument('-no_center', dest='center', action='store_false')
parser.set_defaults(center=False)
parser.add_argument('-downsample', type=int, default=1)
parser.add_argument('-classicnorm', dest='classicnorm', action='store_true')
parser.add_argument(
    '-no_classicnorm', dest='classicnorm', action='store_false')
parser.set_defaults(classicnorm=True)
parser.add_argument('-ampl', type=int, default=441)
parser.add_argument('-cmscrop', type=int, default=550)
parser.add_argument('-rescale', type=int, default=550)
parser.add_argument('-use_adasum', type=bool, default=False)
parser.add_argument(
    '-gradient_predivide_factor', type=float, default=1.0,
    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('-expnum', type=str, default='111')
# parser.add_argument('-hidden_dim',type=str,
#   default = 'hidden_dim = np.hstack((np.repeat(128, 3),441))')
parser.add_argument(
    '-hidden_dim', type=str,
    default='hidden_dim=np.array([5000,2500,1000,441])')
parser.add_argument(
    '-chidden_dim', type=str,
    default='chidden_dim = np.hstack((96,128,np.repeat(256, 3)))')
parser.add_argument(
    '-kernel_sizes', type=str,
    default='kernel_sizes = np.hstack((7,np.repeat(3, 5)))')
parser.add_argument('-num_input_images', type=int, default=1)
parser.add_argument('-model_name', type=str, default='')
parser.add_argument(
    '-use_pretrained', dest='use_pretrained', action='store_true')
parser.add_argument(
    '-no_use_pretrained', dest='use_pretrained', action='store_false')
parser.set_defaults(use_pretrained=False)
parser.add_argument('-weight_decay', type=float, default=0)
parser.add_argument('-merging_order', type=str, default='')
parser.add_argument('-rand_angle', dest='rand_angle', action='store_true')
parser.add_argument('-no_rand_angle', dest='rand_angle', action='store_false')
parser.set_defaults(rand_angle=False)
parser.add_argument('-specie', type=str, default='598')
parser.add_argument('-num_sam_points', type=int, default=500)
parser.add_argument('-loss_between', type=str, default='f')
parser.add_argument('-expdescr', type=str, default='')
parser.add_argument(
    '-use_existing_csv', dest='use_existing_csv', action='store_true')
parser.add_argument(
    '-not_use_existing_csv', dest='use_existing_csv', action='store_false')
parser.set_defaults(use_existing_csv=True)
parser.add_argument('-noise_input', dest='noise_input', action='store_true')
parser.set_defaults(noise_input=False)
parser.add_argument('-haf', dest='haf', action='store_true')
parser.add_argument('-no_haf', dest='haf', action='store_false')
parser.set_defaults(haf=True)
parser.add_argument('-inputt', type=str, default='img')
parser.add_argument('-minmax_f', dest='minmax_f', action='store_true')
parser.add_argument('-no_minmax_f', dest='minmax_f', action='store_false')
parser.set_defaults(minmax_f=True)
parser.add_argument('-criterion', type=str, default='L1')
parser.add_argument('-ngpu', type=int, default=4)
parser.add_argument('-parallel', type=str, default='hvd')
parser.add_argument(
    '-feature_extract', dest='feature_extract', action='store_true')
parser.add_argument(
    '-no_feature_extract', dest='feature_extract', action='store_false')
parser.set_defaults(feature_extract=False)

parser.add_argument('-zero_angle', dest='zero_angle', action='store_true')
parser.add_argument('-no_zero_angle', dest='zero_angle', action='store_false')
parser.set_defaults(zero_angle=False)
opt = parser.parse_args()

tstart = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.parallel == 'hvd':
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.set_num_threads(1)
    rank = hvd.rank()

classicnorm = '_prenormalized' if opt.classicnorm else ''

if '619' in opt.specie:
    original_h, original_w = 2048, 2448
else:
    original_h, original_w = 1000, 1800
os.chdir('/p/home/jusers/cherepashkin1/jureca/circles/finetune_test')
exec('from experiments.'+opt.expnum+'.'+'cnet import *')


def my_loss(output, target):
    myloss = torch.mean(torch.multiply(weightv, (output - target))**2)
    return myloss


mainpath = '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/phenoseed_csv'


# def memuse():
#     for obj in gc.get_objects():
#         try:
#             if (torch.is_tensor(obj) or
#                     (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
#                 print(type(obj), obj.size())
#         except:
#             pass


def train_model(model, dataloaders, samplers, criterion, optimizer, tmean,
                num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
#     train_sampler.set_epoch(epoch)
#    best_model_wts = copy.deepcopy(model.state_dict())
#    best_acc = 0.0
    lossar = np.zeros([2, num_epochs])
    y_n = np.genfromtxt(os.path.join(
        '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/phenoseed_csv',
        'Y_N_'+str(opt.num_sam_points)+classicnorm+'.csv').replace('\\', '/'),
        delimiter=',') 
    y_n = torch.Tensor(y_n).cuda()
    tmeant = torch.Tensor(tmean).cuda()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
#             running_corrects = 0
            rloss = 0.0
            if opt.parallel == 'hvd':
                train_sampler = samplers[phase]
                train_sampler.set_epoch(epoch)
            # Iterate over data.
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                if i_batch == bn:
                    break
                f_n = sample_batched['f_n']
                far = sample_batched['far']
                if opt.inputt == 'img' or opt.inputt == 'pc':
                    inputs = sample_batched['image']
#                     far = sample_batched['far']
                elif opt.inputt == 'f':
                    inputs = far
#                 memuse()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if opt.inputt == 'pc':
                        x = inputs
                        y = torch.zeros([x.shape[0], x.shape[1]])
                        for i in range(x.shape[0]):
                            y[i, :] = torch.sqrt(
                                x[i, :, 0]**2 + x[i, :, 1]**2 + x[i, :, 2]**2)
                        y = torch.unsqueeze(y, 2)
                        y = y.cuda()
                        outputs = model(y)
                    elif opt.inputt == 'img' or opt.inputt == 'f':
                        outputs = model(inputs)
                    if opt.loss_between == 'f' and opt.minmax:
                        ampl = opt.ampl
                        outputs = torch.multiply(
                            outputs, tmeant[3, :ampl] -
                            tmeant[2, :ampl]) + tmeant[2, :ampl]
                    if opt.loss_between == 'f' or opt.loss_between == 'f+f_n':
                        F = torch.zeros(
                            inputs.shape[0], opt.num_sam_points).cuda()
                        y_n2 = torch.unsqueeze(y_n, axis=0)
                        y_n2 = y_n2.repeat(inputs.shape[0], 1, 1)
                        for i in range(inputs.shape[0]):
                            F[i] = torch.matmul(y_n2[i], outputs[i])
                    elif opt.loss_between == 'f_n':
                        loss = criterion(outputs, f_n)
                    if opt.loss_between == 'f':
                        if opt.criterion == 'L2':
                            loss = torch.sqrt(
                                torch.sum((
                                    (far-F)**2)/opt.bs/opt.num_sam_points))
                        elif opt.criterion == 'L1':
                            loss = torch.sum(
                                torch.abs(far-F))/opt.bs/opt.num_sam_points
                    # elif opt.loss_between == 'f+f_n':
                    #     loss = torch.sqrt(torch.sum(outputs,
                    #         f_n)/opt.bs/opt.num_sam_points) +
                    #               torch.sqrt(torch.sum((
                    #             (far-F)**2)/opt.bs/opt.num_sam_points))
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                rloss += loss.item()
#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if bn < len(dataloaders[phase].dataset):
                ebn = bn
            else:
                ebn = len(dataloaders[phase].dataset)
            if phase == 'train':
                lossar[0][epoch] = rloss/ebn
            else:
                lossar[1][epoch] = rloss/ebn
            print('{} Loss: {:.6f}'.format(phase, rloss/ebn))
#             val_acc_history = []
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model, lossar, time_elapsed

# # Initialize and Reshape the Networks
# Load Data


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
                self.df.iloc[idx, 0].replace('csv', '') +
                '_Surface.ply').replace('\\', '/')
            pcd = open3d.io.read_point_cloud(img_name)
            img = np.asarray(pcd.points)
    #         img = np.genfromtxt(img_name, skip_header = 7, skip_footer = 1)
            img = np.concatenate(
                (img, np.zeros([58014 - img.shape[0], 3])), axis=0)
        elif opt.inputt == 'img':
            nim = opt.num_input_images
            torch.manual_seed(idx)
            if opt.noise_input:
                img = torch.rand(nim, original_h, original_w)
            else:
                if not opt.rand_angle:
                    angles_list = np.array(
                        [10*int(36*i/nim) for i in range(nim)])
                    for i, angle in enumerate(angles_list):
                        newa = angle - 10*int(self.df.iloc[idx, 1])
                        angle = newa if newa >= 0 else newa+360
                        angles_list[i] = angle
                else:
                    angles_list = np.random.choice(
                        np.arange(0, 360, 10), size=nim, replace=False)
                if nim == 1 and opt.zero_angle:
                    angles_list = [0]
                    
                img_name = []

                img = np.zeros([nim, original_h, original_w])
                for i in range(nim):
                    img_name = os.path.join(
                        self.root_dir.replace('phenoseed_csv', 'phenoseed'),
                        self.df.iloc[idx, 0].replace('csv', ''),
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
        fara = np.genfromtxt(os.path.join(self.root_dir, (
            self.df.iloc[idx, 0] +
            '_Far_' + str(opt.num_sam_points) + classicnorm +
            '.csv')).replace('\\', '/'), delimiter=',')
        f_na = np.genfromtxt(os.path.join(self.root_dir, (
            self.df.iloc[idx, 0] +
            '_F_N_' + str(opt.num_sam_points) + classicnorm +
            '.csv')).replace('\\', '/'), delimiter=',')
        far = fara[int(self.df.iloc[idx, 1])]
        f_n = f_na[int(self.df.iloc[idx, 1])]
        sample = {'image': img, 'f_n': f_n, 'far': far}
        if self.transform:
            sample = self.transform(sample)
        return sample


class AmpCrop(object):
    """Crop the label, spherical harmonics amplitude."""

    def __init__(self, ampl):
        self.ampl = ampl

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        if self.ampl == 441:
            f_n = f_n
        else:
            f_n = f_n[:self.ampl]
        return {'image': image,
                'f_n': f_n,
                'far': far}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device):
        # assert isinstance(device, str)
        self.device = device

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        f_n = np.squeeze(f_n)
        return {'image': torch.Tensor(image).cuda(),
                'f_n': torch.Tensor(f_n).cuda(),
                'far': torch.Tensor(far).cuda()}


class Minmax3Dimage(object):
    """Normalize 3D input data to be laying in [0,1]"""
    def __init__(self, minv, maxv):
        self.minv = minv
        self.maxv = maxv

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        image = (image-self.minv) / (self.maxv-self.minv)
        return {'image': image,
                'f_n': f_n,
                'far': far}


class Minmax_f(object):
    """Normalize 3D input data to be laying in [0,1]"""

    def __init__(self, minmax):
        minf = minmax[0]
        maxf = minmax[1]
        self.minf = minf
        self.maxf = maxf

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        far = (far-self.minf)/(self.maxf-self.minf)
        return {'image': image,
                'f_n': f_n,
                'far': far}


class Downsample(object):
    """Downsample the input ply file."""

    def __init__(self, ds):
        self.ds = ds

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        ds_image = image[::self.ds, :]
        return {'image': ds_image,
                'f_n': f_n,
                'far': far}


class Shuffleinput(object):
    """Shuffle the rows of input ply file."""

    def __init__(self, shuffle_seed):
        self.shuffle_seed = shuffle_seed

    def __call__(self, sample):
        np.random.seed(self.shuffle_seed)
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        np.random.shuffle(image)
        return {'image': image,
                'f_n': f_n,
                'far': far}


class Minmax(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self, tmean):
        self.tmean = tmean

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        f_n = ((f_n - np.min(self.tmean[2])) /
               (np.max(self.tmean[3])-np.min(self.tmean[2])))

        return {'image': image,
                'f_n': f_n,
                'far': far}


class Reshape(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self, input_layer):
        self.input_layer = input_layer

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        padval = self.input_layer**2 - image.shape[0]
        if padval >= 0:
            image = np.pad(image, ((0, padval), (0, 0)), mode='constant')
        else:
            image = image[:self.input_layer**2]
        image = np.reshape(image, [3, self.input_layer, self.input_layer])
        return {'image': image,
                'f_n': f_n,
                'far': far}


class Normalize(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        X = image[:, 0]
        Y = image[:, 1]
        Z = image[:, 2]
        C = np.zeros([3, 3])
        C[0, 0] = np.matmul(X, X.transpose())
        C[0, 1] = np.matmul(X, Y.transpose())
        C[0, 2] = np.matmul(X, Z.transpose())
        C[1, 0] = C[0, 1]
        C[1, 1] = np.matmul(Y, Y.transpose())
        C[1, 2] = np.matmul(Y, Z.transpose())
        C[2, 0] = C[0, 2]
        C[2, 1] = C[1, 2]
        C[2, 2] = np.matmul(Z, Z.transpose())
        w, v = LA.eig(C)
        image = np.matmul(v.transpose(), image.transpose()).transpose()
        return {'image': image,
                'f_n': f_n,
                'far': far}


class Center(object):
    """Make the center of masses of point cloud to be in the origin."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        image = image - np.mean(image, axis=0)
        return {'image': image,
                'f_n': f_n,
                'far': far}


class CmsCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        h, w = image[0].shape[0:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        h, w = image[0].shape[0:2]
        new_image = np.zeros([opt.num_input_images, new_h, new_w])
        for i in range(opt.num_input_images):
            img = np.squeeze(image[i])
            img *= 255
#            img = np.squeeze(255-image[i])
            properties = regionprops(
                (img > filters.threshold_otsu(img)).astype(int), img)
            cms = tuple(map(lambda x: int(x), properties[0].centroid))
            tempa = (img[cms[0] - new_h//2: cms[0] + new_h//2,
                     cms[1] - new_w//2: cms[1] + new_w//2]).astype(np.uint8)
            padh = (new_h-tempa.shape[0])//2
            padw = (new_w-tempa.shape[1])//2
            tempb = np.pad(
                tempa, ((padh, new_h-tempa.shape[0]-padh),
                        (padw, new_w-tempa.shape[1]-padw)),
                mode='constant', constant_values=0)
            new_image[i] = tempb
        return {'image': new_image/255,
                'f_n': f_n,
                'far': far}


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        if opt.inputt == 'pc':
            img = image
        else:
            h, w = image[0].shape[0:2]
            if h != self.output_size:
                if isinstance(self.output_size, int):
                    if h > w:
                        new_h, new_w = \
                            self.output_size * h / w, self.output_size
                    else:
                        new_h, new_w = \
                            self.output_size, self.output_size * w / h
                else:
                    new_h, new_w = self.output_size
                new_h, new_w = int(new_h), int(new_w)
                img = np.zeros([opt.num_input_images, new_h, new_w])
                for i in range(opt.num_input_images):
                    img[i] = np.squeeze(transform.resize(image[i, :, :],
                                                         (new_h, new_w)))
                transforms.ToTensor()
            elif h == self.output_size:
                img = image
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'image': img,
                'f_n': f_n,
                'far': far}


class Divide255(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, f_n, far = sample['image'], sample['f_n'], sample['far']
        return {'image': image/255,
                'f_n': f_n,
                'far': far}


def getsize(new_hi, ker, srd):
    pad = (0, 0)
    dil = np.asarray((1, 1))
    return(tuple((np.squeeze((np.asarray(new_hi) + 2*np.asarray(pad) -
                  dil*[np.asarray(ker) - 1] + 1) /
                  np.asarray(srd))).astype(int)))


batch_size = opt.bs
tmean = np.genfromtxt(
    '/p/home/jusers/cherepashkin1/jureca/circles/finetune_test/tmean.csv',
    delimiter=',')
if not opt.noise_input:
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
    exec(
         "data_transforms = {\'train\': transforms.Compose([" +
         minmax + minmax3dimage + normalize + center +
         cmscrop + downsample + minmax_f + rescale +
         "AmpCrop(opt.ampl),ToTensor(device)]),\
         'val': transforms.Compose([" +
         minmax + minmax3dimage + normalize + center +
         cmscrop + downsample + minmax_f + rescale +
         "AmpCrop(opt.ampl),ToTensor(device)])}")
else:
    exec(
         "data_transforms = {\'train\': transforms.Compose([" + rescale +
         "AmpCrop(opt.ampl), ToTensor(device)]),'val': transforms.Compose([" +
         rescale +
         "AmpCrop(opt.ampl),ToTensor(device)])}")
kwargs = {}
if opt.parallel == 'hvd':
    if (
        kwargs.get('num_workers', 0) > 0 and
        hasattr(mp, '_supports_context') and mp._supports_context and
            'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

print("Initializing Datasets and Dataloaders...")

mainpath = '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/phenoseed_csv'
pts = os.path.join(mainpath, 'pathes_to: ' + opt.specie + '.csv')
if os.path.isfile(pts) and opt.use_existing_csv:
    lframe = pd.read_csv(pts)
#     train_part = int(0.8*len(lframe))
#     df_dict = {'train': lframe[:train_part], 'val': lframe[train_part:]}
else:
    cip = []
    for word in opt.specie.split(','):
        csvpath = os.path.join(mainpath, word+'csv')
        for root, directories, filenames in os.walk(csvpath):
            for filename in filenames:
                search_str = 'F_N_' + str(opt.num_sam_points) + '.csv'
                if filename[-len(search_str):] == search_str:
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
train_part = int(0.8*len(lframe))
df_dict = {'train': lframe[:train_part], 'val': lframe[train_part:]}

# Create training and validation datasets
image_datasets = {x: Seed3D_Dataset(
    mainpath, df_dict[x],
    transform=data_transforms[x]) for x in ['train', 'val']}
if opt.parallel == 'hvd':
    samplers = {x: torch.utils.data.distributed.DistributedSampler(
        image_datasets[x], num_replicas=hvd.size(),
        rank=hvd.rank()) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=opt.bs, shuffle=False, sampler=samplers[x],
        num_workers=0, pin_memory=False,
        **kwargs) for x in ['train', 'val']}
elif opt.parallel == 'torch':
    samplers = None
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size, shuffle=False, num_workers=0)
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
else:
    if opt.inputt == 'img':
        smodel = CNet(
            hidden_dim, chidden_dim, kernel_sizes,
            opt.num_input_images, opt.rescale,
            int(opt.rescale*original_w/original_h), opt.haf)
    elif opt.inputt == 'f':
        smodel = CNet(
            hidden_dim, chidden_dim, kernel_sizes,
            opt.num_input_images, 1,
            opt.num_sam_points, opt.haf)
    elif opt.inputt == 'pc':
        smodel = CNet(
            hidden_dim, chidden_dim, kernel_sizes,
            opt.num_input_images, np.floor(58014/opt.downsample).astype(int),
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
model, lossar, time_elapsed = train_model(
    smodel, dataloaders_dict, samplers, criterion, optimizer, tmean,
    num_epochs=num_epochs, is_inception=(opt.model_name == "inception"))
rank = hvd.rank() if opt.parallel == 'hvd' else 0

if rank == 0:
    dir1 = '/p/home/jusers/cherepashkin1/jureca/'+\
        'cherepashkin1/598test/plot_output/'+opt.expnum
    if not os.path.exists(dir1):
        os.mkdir(dir1)
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
    np.save(dirname+'lossar.npy', lossar)

    def lossfig(lossar, ylabel):
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
    lossfig(lossar, 'Loss')
    lossfig(np.log10(lossar), 'log10(Loss)')

    file = open(dirname+"job-parameters.txt", "w")
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
        'time_elapsed=' +
        str(datetime.timedelta(seconds=time_elapsed)) + '\n' +
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
        'num_input_images=' + str(opt.num_input_images) + '\n' +
        'merging_order=' + str(opt.merging_order) + '\n' +
        'specie=' + opt.specie + '\n' +
        'number of sampled points=' + str(opt.num_sam_points) + '\n' +
        'loss is between model output and gt of =' +
        opt.loss_between + '\n' +
        'rand_angle=' + str(opt.rand_angle) + '\n' +
        'ngpu=' + str(opt.ngpu) + '\n' +
        'was there just noise as input? =' + str(opt.noise_input))
    file.write(parameters_list)
    file.close()

    original_stdout = sys.stdout
    with open(dirname+"job-parameters.txt", 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(smodel)
        # Reset the standard output to its original value
        sys.stdout = original_stdout
    torch.save(model.state_dict(), dirname+"model")
    for x in ['train', 'val']:
        df_dict[x].to_csv(dirname+'pathes_df_'+x+'.csv')
    print(time.time()-tstart)
    # ## Return SH coefficient vector from the trained model
    phase = 'val'
    bn = 1
    dataloaders = dataloaders_dict
    for i_batch, sample_batched in enumerate(dataloaders[phase]):
        if i_batch == bn:
            break
        if opt.inputt == 'f':
            inputs = sample_batched['far']
        elif opt.inputt == 'img' or opt.inputt == 'pc':
            inputs = sample_batched['image']
        if opt.inputt == 'pc':
            x = inputs
            y = torch.zeros([x.shape[0], x.shape[1]])
            y = y.cuda()
            for i in range(x.shape[0]):
                y[i, :] = torch.sqrt(
                    x[i, :, 0]**2 + x[i, :, 1]**2 + x[i, :, 2]**2)
            y = torch.unsqueeze(y, 2)
            o = model(y)
        else:
            o = model(inputs)
        gt = sample_batched['f_n']

    if opt.minmax:
        ampl = opt.ampl
        output = np.multiply(
            model(inputs).detach().cpu().numpy(),
            tmean[3, :ampl]-tmean[2, :ampl])+tmean[2, :ampl]
        real_output = np.multiply(
            gt.detach().cpu().numpy(),
            tmean[3, :ampl]-tmean[2, :ampl])+tmean[2, :ampl]
    else:
        output = o.detach().cpu().numpy()
        real_output = gt.detach().cpu().numpy()
    np.savetxt(dirname+'o', output, delimiter=',')
    np.savetxt(dirname+'gt', real_output, delimiter=',')
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

    def plot_Y(ax, coef):
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
        print(Yz.shape)
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
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    Yx, Yy, Yz = plot_Y(ax, output[0])
    plt.savefig(dirname + 'surface')
