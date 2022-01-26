#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# In[3]:
from __future__ import print_function
from __future__ import division
import argparse
import open3d 
import torch
import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from skimage import io, transform, data
from skimage.measure import regionprops
from skimage import filters
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.special import sph_harm
import sys
import shutil
from numpy import linalg as LA

# In[4]:
from cnet import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# In[5]:
if torch.cuda.device_count()>1:
    device = torch.device('cuda:0')
elif torch.cuda.device_count()==1:
    device = torch.device('cuda')
else: 
    device = torch.device('cpu')
print(device)
# In[6]:
parser = argparse.ArgumentParser()
parser.add_argument('-bs', type=int, default=4)
parser.add_argument('-epoch', type=int, default=8)
parser.add_argument('-bn', type=int, default=8)
parser.add_argument('-lr', type=float, default=5e-5)
parser.add_argument('-dataparallel', type=bool, default=True)
parser.add_argument('-minmax', dest='minmax', action='store_true')
parser.add_argument('-no_minmax', dest='minmax', action='store_false')
parser.set_defaults(minmax=False)
parser.add_argument('-minmax3dimage', dest='minmax3dimage', action='store_true')
parser.add_argument('-no_minmax3dimage', dest='minmax3dimage', action='store_false')
parser.set_defaults(minmax3dimage=False)
parser.add_argument('-normalize', dest='normalize', action='store_true')
parser.add_argument('-no_normalize', dest='normalize', action='store_false')
parser.set_defaults(normalize=False)
parser.add_argument('-center', dest='center', action='store_true')
parser.add_argument('-no_center', dest='center', action='store_false')
parser.set_defaults(center=False)
parser.add_argument('-downsample', type=int, default=1)
parser.add_argument('-classicnorm', dest='classicnorm', action='store_true')
parser.add_argument('-no_classicnorm', dest='classicnorm', action='store_false')
parser.set_defaults(classicnorm=False)
parser.add_argument('-ampl', type=int, default=441)
parser.add_argument('-cmscrop', type=int, default=0)
parser.add_argument('-rescale', type=int, default=225)
parser.add_argument('-use_adasum',type=bool, default = False)
parser.add_argument('-gradient_predivide_factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('-expnum',type=str, default = '111')
parser.add_argument('-hidden_dim',type=str, default = 'hidden_dim = np.hstack((np.repeat(128, 3),441))')
parser.add_argument('-chidden_dim',type=str, default = 'chidden_dim = np.hstack((96,128,np.repeat(256, 3)))')
parser.add_argument('-kernel_sizes',type=str, default = 'kernel_sizes = np.hstack((7,np.repeat(3, 5)))')
parser.add_argument('-input3d', dest='inputt', action='store_true')
parser.add_argument('-input2d', dest='inputt', action='store_false')
parser.set_defaults(inputt=False)
parser.add_argument('-num_input_images', type=int, default=3)
parser.add_argument('-model_name', type=str, default='')
parser.add_argument('-use_pretrained', dest='use_pretrained', action='store_true')
parser.add_argument('-no_use_pretrained', dest='use_pretrained', action='store_false')
parser.set_defaults(use_pretrained=False)
parser.add_argument('-weight_decay', type=float, default=0)
opt = parser.parse_args()
tstart = time.time()
hvd.init()
torch.cuda.set_device(hvd.local_rank())
torch.set_num_threads(1)

if opt.classicnorm: 
    datatype = '1000'
else:
    datatype = '1001'
# In[8]:
# Helper Functions - train_model
# In[9]:
def my_loss(output, target):
    l = torch.mean(torch.multiply(weightv,(output - target))**2)
    return l
Y_N = np.genfromtxt('1491988Y_N.csv',delimiter=',')
D = np.genfromtxt('1491988bX.csv',delimiter=',')
F = np.matmul(Y_N,coef)
x = np.zeros([3,1000])
x[0,:] = F * np.cos(D[0,:]) * np.sin(D[1,:])
x[1,:] = F * np.sin(D[0,:]) * np.sin(D[1,:])
x[2,:] = F * np.cos(D[1,:])
def train_model(model, dataloaders, samplers, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
#    train_sampler.set_epoch(epoch)
#    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    lossar = np.zeros([2,num_epochs])
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
            running_corrects = 0
            rloss = 0.0
            train_sampler = samplers[phase]
            train_sampler.set_epoch(epoch)            
            # Iterate over data.
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                if i_batch == bn:
                    break
                inputs = sample_batched['image']
                labels = sample_batched['landmarks']
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        if opt.inputt:
                            x = inputs
                            y = torch.zeros([x.shape[0], x.shape[1]])
                            for i in range(x.shape[0]):
                                y[i,:] = torch.sqrt(x[i,:,0]*x[i,:,0]+x[i,:,1]*x[i,:,1]+x[i,:,2]*x[i,:,2])
                            y = torch.unsqueeze(y,2)
                            y = y.cuda()
                            outputs = model(y)
                        else:
                            outputs = model(inputs)
                        loss = criterion(outputs, labels)
#                         F = torch.matmul(outputs,Y_N2)
                        #loss = my_loss(outputs,labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                rloss += loss.item()
#                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if bn < len(dataloaders[phase].dataset):
                ebn = bn
            else:
                ebn = len(dataloaders[phase].dataset)
            if phase == 'train':
                lossar[0][epoch] = rloss/ebn
            else: 
                lossar[1][epoch] = rloss/ebn          
#            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.6f}'.format(phase, rloss/ebn))
#            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,epoch_acc))
            val_acc_history = []
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, lossar, time_elapsed


# # Initialize and Reshape the Networks
# In[10]:
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size
# Load Data
# In[11]:
class Seed3D_Dataset(Dataset):
    """seed point cloud dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
#        self.landmarks_frame = pd.read_csv(os.path.join(root_dir,'F_N_1001.csv'))
        self.landmarks_frame = pd.read_csv(os.path.join(root_dir,'F_N_'+ datatype + '.csv'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rd = self.root_dir.replace('598test','598')
        if opt.inputt:
            img_name =         os.path.join(rd, self.landmarks_frame.iloc[idx, 0]+'_Surface.ply').replace('\\','/')
            pcd = open3d.io.read_point_cloud(img_name)
            img = np.asarray(pcd.points)
    #         img = np.genfromtxt(img_name, skip_header = 7, skip_footer = 1)
            img = np.concatenate((img, np.zeros([58014-img.shape[0],3])), axis=0)
        else:
            nim = opt.num_input_images
            img = np.zeros([nim,1000,1800,1])
            for i in range(nim):
                img_name = \
                os.path.join(rd,\
                             self.landmarks_frame.iloc[idx, 0],\
                'rotation_'+str(10*int(36*i/nim)).zfill(3)+'.tif').replace('\\','/')
                img[i] = np.expand_dims(np.asarray(io.imread(img_name)), axis=2)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 441)
        y_n = genfromtxt(os.path.join(rd, self.landmarks_frame.iloc[idx, 0]+'Y_N.csv').replace('\\','/'),delimiter=',')
        directions = genfromtxt(os.path.join(rd, self.landmarks_frame.iloc[idx, 0]+'bX.csv').replace('\\','/'),delimiter=',')
        sample = {'image': img, 'landmarks': landmarks, 'y_n': y_n, 'directions': directions}
        if self.transform:
            sample = self.transform(sample)
        return sample
# In[12]:
# In[13]:
class AmpCrop(object):
    """Crop the label, spherical harmonics amplitude."""
    def __init__(self, ampl):
        self.ampl = ampl
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        return {'image': image,
                'landmarks': landmarks[:,:self.ampl]}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device):
#        assert isinstance(device, str)
        self.device = device
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        landmarks = np.squeeze(landmarks)
        return {'image': torch.Tensor(image).cuda(),
                'landmarks': torch.Tensor(landmarks).cuda()}
#         return {'image': torch.Tensor(image).to(self.device),
#                 'landmarks': torch.Tensor(landmarks).to(self.device)}
class Minmax3Dimage(object):
    """Normalize 3D input data to be laying in [0,1]"""
    def __init__(self,minmax):
        minf = minmax[0]
        maxf = minmax[1]
        self.minf = minf
        self.maxf = maxf
#        assert isinstance(device, str)
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = (image-self.minf)/(self.maxf-self.minf)
        #         for i in range(3):
#             image[:,i] = (image[:,i]-np.min(image,axis=0)[i])/\
#             (np.max(image,axis=0)[i]-np.min(image,axis=0)[i])
        return {'image': image,
                'landmarks': landmarks}

class Downsample(object):
    """Downsample the input ply file."""
    def __init__(self, ds):
        #assert isinstance(output_size, (int, tuple))
        self.ds = ds
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        ds_image = image[::self.ds,:]
        return {'image': ds_image,
                'landmarks': landmarks}
class Shuffleinput(object):
    """Shuffle the rows of input ply file."""
    def __init__(self, shuffle_seed):
        #assert isinstance(output_size, (int, tuple))
        self.shuffle_seed = shuffle_seed
    def __call__(self, sample):
        np.random.seed(self.shuffle_seed)
        image, landmarks = sample['image'], sample['landmarks']
        np.random.shuffle(image) 
        return {'image': image,
                'landmarks': landmarks}
class Minmax(object):
    """Normalize the input data to lay in [0,1]."""
    def __init__(self, tmean):
#        assert isinstance(tmean, numpy.ndarray)
        self.tmean = tmean
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
#        print(type(landmarks))
#        print(type(self.tmean[3]))
#        landmarks = (landmarks - self.tmean[2])/(self.tmean[3]-self.tmean[2])
        landmarks = (landmarks - np.min(self.tmean[2]))/(np.max(self.tmean[3])-np.min(self.tmean[2]))
        
        return {'image': image,
                'landmarks': landmarks}
class Reshape(object):
    """Normalize the input data to lay in [0,1]."""
    def __init__(self, input_layer):
#        assert isinstance(tmean, numpy.ndarray)
        self.input_layer = input_layer
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
#        print(type(landmarks))
#        print(type(self.tmean[3]))
        padval = self.input_layer**2-image.shape[0]
        if padval >= 0:
            image = np.pad(image, ((0,padval),(0,0)), mode='constant')
        else: 
            image = image[:self.input_layer**2]
        image = np.reshape(image, [3,self.input_layer,self.input_layer])
#        landmarks = (landmarks - self.tmean[2])/(self.tmean[3]-self.tmean[2]) 
        return {'image': image,
                'landmarks': landmarks}
class Normalize(object):
    """Normalize the input data to lay in [0,1]."""
    def __init__(self):
#        assert isinstance(tmean, numpy.ndarray)
        a = 1
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
#        print(type(landmarks))
#        print(type(self.tmean[3]))
        X = image[:,0]
        Y = image[:,1]
        Z = image[:,2]
        C = np.zeros([3,3])
        C[0,0] = np.matmul(X,X.transpose())
        C[0,1] = np.matmul(X,Y.transpose())
        C[0,2] = np.matmul(X,Z.transpose())
        C[1,0] = C[0,1]
        C[1,1] = np.matmul(Y,Y.transpose())
        C[1,2] = np.matmul(Y,Z.transpose())
        C[2,0] = C[0,2]
        C[2,1] = C[1,2]
        C[2,2] = np.matmul(Z,Z.transpose())
        w,v = LA.eig(C)
        image = np.matmul(v.transpose(),image.transpose()).transpose()
        return {'image': image,
                'landmarks': landmarks}
class Center(object):
    """Normalize the input data to lay in [0,1]."""
    def __init__(self):
#        assert isinstance(tmean, numpy.ndarray)
        a = 1
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image - np.mean(image,axis = 0)
        return {'image': image,
                'landmarks': landmarks}
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
        image, landmarks = sample['image'], sample['landmarks']

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
        new_image = np.zeros([3,new_h, new_w])
        for i in range(3):
            img = np.squeeze(255-image[i])
            properties = regionprops((img > filters.threshold_otsu(img)).astype(int), img)
            cms = tuple(map(lambda x: int(x), properties[0].centroid))
            tempa = (255-img[cms[0]-new_h//2:cms[0]+new_h//2,\
                                  cms[1]-new_w//2:cms[1]+new_w//2]).astype(np.uint8)
            padh = (new_h-tempa.shape[0])//2
            padw = (new_w-tempa.shape[1])//2
            tempb = np.pad(tempa, \
                ((padh, new_h-tempa.shape[0]-padh),(padw,new_w-tempa.shape[1]-padw)),\
                mode='constant', constant_values = 255)
            new_image[i] = tempb
        return {'image': new_image/255, 'landmarks': landmarks}
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
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image[0].shape[0:2]
        if h != self.output_size:
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)
            img = np.zeros([opt.num_input_images,new_h, new_w])
    #        print(img.shape)
            for i in range(opt.num_input_images):
                img[i] = np.squeeze(transform.resize(image[i], (new_h, new_w)))
            transforms.ToTensor()
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'image': img, 'landmarks': landmarks}    
class Divide255(object):
    """Normalize the input data to lay in [0,1]."""
    def __init__(self):
        a = 1
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        return {'image': image/255,
                'landmarks': landmarks}
# In[14]:
def getsize(new_hi,ker,srd): 
    pad = (0,0)
    dil = np.asarray((1,1))
    return(tuple((np.squeeze((np.asarray(new_hi)+    2*np.asarray(pad)-dil*[np.asarray(ker)-1]+1)/                             np.asarray(srd))).astype(int)))
# In[18]:
batch_size = opt.bs
data_dir = 'D:/seva/598test'
data_dir = '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/598test'
model_name = "densenet"
#ds = 5
tmean = np.genfromtxt('/p/home/jusers/cherepashkin1/jureca/circles/finetune_test/tmean.csv', delimiter = ',')
minmax,minmax3dimage,normalize,center=['']*4
if opt.minmax:
    minmax = 'Minmax(tmean[:,:opt.ampl]), '
else: 
    minmax = ''  
if opt.minmax3dimage:
    minmax3dimage = 'Minmax3Dimage((0.60117054538415,110.972068294924)), '
else: 
    minmax3dimage = '' 
if opt.normalize:
    normalize = 'Normalize(), '
else: 
    normalize = ''
if opt.center:
    center = 'Center(), '
else: 
    center = ''
if opt.cmscrop:
    cmscrop = 'CmsCrop(opt.cmscrop),'
else: 
    cmscrop = ''    
exec("data_transforms = {\'train\': transforms.Compose(["+\
     minmax+minmax3dimage+normalize+center+cmscrop+\
     "AmpCrop(opt.ampl),Downsample(opt.downsample),Rescale(opt.rescale),Divide255(),ToTensor(device)]),\
     'val': transforms.Compose(["+\
     minmax+minmax3dimage+normalize+center+cmscrop+\
     "AmpCrop(opt.ampl),Downsample(opt.downsample),Rescale(opt.rescale),Divide255(),ToTensor(device)])}")


# data_transforms = {
#     'train': transforms.Compose([
# #             AmpCrop(ampl),\
# #             Minmax(tmean[:,:ampl]),\
#             Minmax3Dimage((0.60117054538415,110.972068294924)),\
#            Downsample(ds),\
# #            Center(),\
# #           Shuffleinput(0),\
# #             Normalize(),\
# #             Reshape(224),\
#             ToTensor(device)
#     ]),
#     'val': transforms.Compose([
# #             AmpCrop(ampl),\
# #             Minmax(tmean[:,:ampl]),\
#             Minmax3Dimage((0.60117054538415,110.972068294924)),\
#            Downsample(ds),\
# #            Center(),\
# #           Shuffleinput(0),\
# #             Normalize(),\
# #             Reshape(224),\
#             ToTensor(device)
#     ]),
# }
kwargs={}
if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: Seed3D_Dataset(root_dir=os.path.join(data_dir,x),                                     transform=data_transforms[x]) 
                  for x in ['train', 'val']}
samplers = {x: torch.utils.data.distributed.DistributedSampler(image_datasets[x],    num_replicas=hvd.size(), rank=hvd.rank()) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.bs, shuffle=False, sampler=samplers[x], num_workers = 0, pin_memory = False, **kwargs) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset = Seed3D_Dataset(csv_file=mainpath+'/sh_paramters.csv', root_dir=data_dir, transform=data_transform)
# dataloader = DataLoader(dataset, bs,
#                         shuffle=False, num_workers=0)
# Create the Optimizer
# In[19]:
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

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


# Run Training and Validation Step
# In[20]:
# Setup the loss fxn
criterion = nn.MSELoss(reduction='mean')
bn = opt.bn
num_epochs = opt.epoch
# model_name = 'densenet'
lossar = np.zeros([2,num_epochs])
# Train and evaluate
exec(opt.hidden_dim)
exec(opt.chidden_dim)
exec(opt.kernel_sizes)
if bool(opt.model_name): 
    smodel, input_size = initialize_model(opt.model_name, opt.ampl, feature_extract, use_pretrained=opt.use_pretrained)
else:
    smodel = CNet(hidden_dim,chidden_dim,kernel_sizes,opt.num_input_images, opt.rescale)
#if opt.dataparallel:
    #smodel = nn.DataParallel(smodel)
lr=opt.lr*hvd.local_size()
smodel.cuda()
#smodel.to(device)
optimizer = torch.optim.Adam(smodel.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay,             amsgrad=False)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(smodel.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.none
#compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=smodel.named_parameters(),
                                     compression=compression,
                                     op=hvd.Adasum if opt.use_adasum else hvd.Average,    gradient_predivide_factor=opt.gradient_predivide_factor)

model, lossar, time_elapsed = train_model(smodel, dataloaders_dict, samplers, criterion, optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))


# In[24]:

rank = hvd.rank()

if rank == 0:
    dir1 = '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/598test/plot_output/'+opt.expnum
    #dirname = '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/598test/plot_output/'+opt.expnum+'/'+str(int(time.time()))+'/'
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    i=0
    while True:
        n = str(i)
        dirname = dir1+'/'+n.zfill(3)+'/'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            break
        else:
            i+=1
            continue
    np.save(dirname+'lossar.npy',lossar)
    # In[25]:
    plt.rcParams["figure.figsize"] = (9,5)
    fig = plt.figure()
    axes = plt.gca()
    # ymin, ymax = , 2
    # axes.set_ylim([ymin,ymax])
    labels_text = ['train', 'val']
    for i in range(2):
        plt.plot(np.arange(lossar.shape[1]),lossar[i,:],label=labels_text[i],linewidth=3)
    axes.set_xticks(np.arange(0, int(num_epochs*1.1)),                          max(int(num_epochs*0.1),1))
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.,            fontsize = 24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.autoscale(enable=True, axis='both', tight=None)
    # ymin, ymax = 0,2
    # axes.set_ylim([ymin,ymax])
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('MSE Loss', fontsize=24)
    plt.savefig(dirname+#                     'num_ampl='+str(ampl)+\
                    'bn'+str(bn)+\
                    'bs'+str(opt.bs)+\
                    'ds'+str(opt.downsample)+\
                    'epochs'+str(num_epochs)+\
                    'lr'+str(lr)+\
                    't'+str(time.time())+\
                '.png', bbox_inches='tight')


    # ## Return SH coefficient vector from the trained model
    # In[18]:
    phase = 'val'
    bn = 1
#    ampl = 441
    dataloaders = dataloaders_dict
    for i_batch, sample_batched in enumerate(dataloaders[phase]):
        if i_batch == bn:
            break
        inputs = sample_batched['image']
        if opt.inputt:
            x = inputs
            y = torch.zeros([x.shape[0], x.shape[1]])
            y = y.cuda()
            for i in range(x.shape[0]):
                y[i,:] = torch.sqrt(x[i,:,0]*x[i,:,0]+x[i,:,1]*x[i,:,1]+x[i,:,2]*x[i,:,2])
            y = torch.unsqueeze(y,2)
            o = model(y)
        else:
            o = model(inputs)
        gt = sample_batched['landmarks']



    if opt.minmax:
        ampl = opt.ampl
        output = np.multiply(model(inputs).detach().cpu().numpy(),tmean[3,:ampl]-tmean[2,:ampl])+tmean[2,:ampl]
        real_output = np.multiply(gt.detach().cpu().numpy(),tmean[3,:ampl]-tmean[2,:ampl])+tmean[2,:ampl]
    else:
        output = o.detach().cpu().numpy()
        real_output = gt.detach().cpu().numpy()
    # In[19]:
    # In[ ]:
    np.savetxt(dirname+'o', output, delimiter = ',')
    np.savetxt(dirname+'gt', real_output, delimiter = ',')
    # In[ ]:
    plt.rc('text', usetex=True)
    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    #r = np.linspace(-1, 1, 100)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([np.sin(theta) * np.sin(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta)])

    # In[ ]:
    #coef = np.random.rand(10,10)
    def plot_Y(ax, coef):
        """Plot the spherical harmonic of degree el and order m on Axes ax."""
        f = np.zeros([100,100]).astype('complex128')
        for l in range(int(np.sqrt(len(coef)))):
            for m in range(-l,l+1):
                fb = coef[l*(l+1)+m] * sph_harm(abs(m), l, phi, theta)
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
        ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
        ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
        ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
        # Set the Axes limits and title, turn off the Axes frame.
    #    ax.set_title(r'$Y_{{{},{}}}$'.format(el, m))
        ax_lim = 40
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_zlim(-ax_lim, ax_lim)
        ax.axis('on')
        return(Yx, Yy, Yz)
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    #l, m = 2, 0
    # coef = [116.40520943, -4.51331063,  -1.22070776, 2.53049672, 1.25615844,\
    #    1.59720443,  28.4781262,   -0.33878004,  -4.02484515]
    Yx, Yy, Yz = plot_Y(ax, output[0])
    #plot_Y(ax,6,3)
    #plt.savefig('Y{}_{}.png'.format(l, m))
    #plt.show()
    plt.savefig(dirname+'surface')

    file = open(dirname+"job-parameters.txt", "w") 
    file.write('epoch='+str(opt.epoch)+'\n'\
                'bn='+str(opt.bn)+'\n'\
                'bs='+str(opt.bs)+'\n'\
                'lr='+str(opt.lr)+'\n'\
                'dataparallel='+str(opt.dataparallel)+'\n'\
#                 'ampcrop='+str(opt.ampcrop)+'\n'\
                'minmax='+str(opt.minmax)+'\n'\
                'minmax3dimage='+str(opt.minmax3dimage)+'\n'\
                'downsample='+str(opt.downsample)+'\n'\
                'normalize='+str(opt.normalize)+'\n'\
                'center='+str(opt.center)+'\n'\
                'classicnorm='+str(opt.classicnorm)+'\n'\
                'ampl='+str(opt.ampl)+'\n'\
                'time_elapsed='+str(time_elapsed)+'\n'\
                'hidden_dim='+opt.hidden_dim+'\n'\
                'chidden_dim='+opt.chidden_dim+'\n'\
                'model_name='+opt.model_name+'\n'\
                'weight_decay='+str(opt.weight_decay)+'\n'\
                'num_input_images='+str(opt.num_input_images)+'\n'
              ) 
    file.close() 
    
    original_stdout = sys.stdout  
    with open(dirname+"job-parameters.txt", 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(smodel)
        sys.stdout = original_stdout # Reset the standard output to its original value
    # In[ ]:
    shutil.copyfile('/p/home/jusers/cherepashkin1/jureca/circles/finetune_test/cnet.py', dirname+"cnet.py")
    torch.save(model.state_dict(), dirname+"model")
    print(time.time()-tstart)
# In[ ]:




