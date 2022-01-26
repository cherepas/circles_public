#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')


#
# Finetuning Torchvision Models
# =============================
#
# **Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__
#
#
#

# In this tutorial we will take a deeper look at how to finetune and
# feature extract the `torchvision
# models <https://pytorch.org/docs/stable/torchvision/models.html>`__, all
# of which have been pretrained on the 1000-class Imagenet dataset. This
# tutorial will give an indepth look at how to work with several modern
# CNN architectures, and will build an intuition for finetuning any
# PyTorch model. Since each model architecture is different, there is no
# boilerplate finetuning code that will work in all scenarios. Rather, the
# researcher must look at the existing architecture and make custom
# adjustments for each model.
#
# In this document we will perform two types of transfer learning:
# finetuning and feature extraction. In **finetuning**, we start with a
# pretrained model and update *all* of the model’s parameters for our new
# task, in essence retraining the whole model. In **feature extraction**,
# we start with a pretrained model and only update the final layer weights
# from which we derive predictions. It is called feature extraction
# because we use the pretrained CNN as a fixed feature-extractor, and only
# change the output layer. For more technical information about transfer
# learning see `here <https://cs231n.github.io/transfer-learning/>`__ and
# `here <https://ruder.io/transfer-learning/>`__.
#
# In general both transfer learning methods follow the same few steps:
#
# -  Initialize the pretrained model
# -  Reshape the final layer(s) to have the same number of outputs as the
#    number of classes in the new dataset
# -  Define for the optimization algorithm which parameters we want to
#    update during training
# -  Run the training step
#
#
#

# In[2]:


from __future__ import print_function
from __future__ import division
import torch
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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# In[3]:


if torch.cuda.device_count()>1:
    device = torch.device('cuda:2')
elif torch.cuda.device_count()==1:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


# Inputs
# ------
#
# Here are all of the parameters to change for the run. We will use the
# *hymenoptera_data* dataset which can be downloaded
# `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__.
# This dataset contains two classes, **bees** and **ants**, and is
# structured such that we can use the
# `ImageFolder <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder>`__
# dataset, rather than writing our own custom dataset. Download the data
# and set the ``data_dir`` input to the root directory of the dataset. The
# ``model_name`` input is the name of the model you wish to use and must
# be selected from this list:
#
# ::
#
#    [resnet, alexnet, vgg, squeezenet, densenet, inception]
#
# The other inputs are as follows: ``num_classes`` is the number of
# classes in the dataset, ``batch_size`` is the batch size used for
# training and may be adjusted according to the capability of your
# machine, ``num_epochs`` is the number of training epochs we want to run,
# and ``feature_extract`` is a boolean that defines if we are finetuning
# or feature extracting. If ``feature_extract = False``, the model is
# finetuned and all model parameters are updated. If
# ``feature_extract = True``, only the last layer parameters are updated,
# the others remain fixed.
#
#
#

# In[4]:


# mpl = [r'D:\seva\hymenoptera_data',  r'D:/data/hymenoptera_data']
# mpl = [s.replace('\\','/') for s in mpl]
# i = 0
# while not os.path.exists(mpl[i]):
#     print(i)
#     i+=1
# data_dir = mpl[i]
# print(data_dir)


# In[5]:


depth = 2**8-1.
D_out = 441
C_in = 3 #how many views of one seed
mpl = [r'/home/cherepashkin/data/598_processing',  r'D:\data\seeds\598', r'D:\seva\598_processing'.replace('\\','/')]
mpl = [s.replace('\\','/') for s in mpl]
i = 0
while not os.path.exists(mpl[i]):
    print(i)
    i+=1
data_dir = mpl[i]
print(data_dir)


# In[6]:


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = D:\seva\hymenoptera_data
#    "D:/data/hymenoptera_data"
    #""./data/hymenoptera_data"
data_dir

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

#number of spherical harmonics amplituds to regress
ampl = 16

# Number of classes in the dataset
num_classes = ampl

#downsample ply, taking point every ds steps
ds = 10


# Helper Functions - train_model
# ----------------
#
# Before we write the code for adjusting the models, lets define a few
# helper functions.
#
# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for, and a boolean flag for when the model is an
# Inception model. The *is_inception* flag is used to accomodate the
# *Inception v3* model, as that architecture uses an auxiliary output and
# the overall model loss respects both the auxiliary output and the final
# output, as described
# `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
#
#
#

# In[7]:


#print(tmean[:,:ampl])


# In[ ]:


#weightv = torch.tensor([0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]).to(device)


# In[8]:


def my_loss(output, target):
    l = torch.mean(torch.multiply(weightv,(output - target))**2)
    return l


# In[9]:


#my_loss(torch.zeros([8,9]),torch.zeros([8,9]))


# In[62]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs, bs, device, is_inception=False):
    since = time.time()

    val_acc_history = []

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

            # Iterate over data.
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
#                 print(len(dataloaders[phase]))
                if i_batch == bn:
                    break
                inputs = sample_batched['image']
                labels = sample_batched['landmarks']
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
#                         print(inputs.shape)
                        F = torch.sum(torch.mul(inputs,inputs), axis = 0)
#                         print(F.shape)
#                         x = torch.zeros([8, 1161, 3])
                        x = inputs
#                         print(x.shape)
                        y = torch.zeros([x.shape[0], x.shape[1]])
                        y = y.to(device)
                        for i in range(x.shape[0]):
                            y[i,:] = torch.sqrt(x[i,:,0]*x[i,:,0]+x[i,:,1]*x[i,:,1]+x[i,:,2]*x[i,:,2])
                        y = torch.unsqueeze(y,2)
                        outputs = model(y)
                        # self-supervised part

                        loss = criterion(outputs, labels)
#                         F = torch.matmul(outputs,Y_N2)
#                         loss = criterion(, inputs)
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

            # deep copy the model
#            if phase == 'val' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#            best_model_wts = copy.deepcopy(model.state_dict())
#             if phase == 'val':
#                 val_acc_history.append(epoch_acc)
            val_acc_history = []

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
#    model.load_state_dict(best_model_wts)
    return model, lossar


# In[11]:


#inputs.shape


# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
#
#
#

# In[12]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# # Initialize and Reshape the Networks

# In[13]:


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


# In[14]:


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
#print(model_ft)


# ## search of the min max of the sh amplitudes

# In[15]:


# mainpath = data_dir + '\train'
# D_out = 441
# if not(os.path.isfile('tmean.csv') and os.path.isfile(mainpath+'/sh.csv')):
#     cip = []
#     for root, directories, filenames in os.walk(mainpath):
#         for filename in filenames:
#             if filename[-8:] == 'F_20.csv':
#                 cip.append(os.path.join(root,filename))
#     labels = np.zeros([len(cip),D_out])
#     for i in range(len(cip)):
#         labels[i,:] = torch.tensor(np.genfromtxt(cip[i], delimiter='\n'))
#     tmean = np.zeros([4,D_out])
#     tmean[0,:] = np.mean(labels, axis = 0)
#     tmean[1,:] = np.std(labels, axis = 0)
#     tmean[2,:] = np.min(labels, axis = 0)
#     tmean[3,:] = np.max(labels, axis = 0)
#     numpy.savetxt("tmean.csv", tmean, delimiter=",")
#     landmarks_frame = pd.DataFrame(data=labels,index=range(len(cip)),\
#                                    columns=['f'+str(i) for i in range(D_out)])
#     landmarks_frame.insert(0, 'file_name', \
#                            [cip[i][-24:-9] for i in range(len(cip))])
#     landmarks_frame.to_csv(mainpath+'/sh_paramters.csv', index=False)


# In[16]:


tmean = np.genfromtxt("tmean.csv", delimiter=',')


# Load Data
# ---------
#
# Now that we know what the input size must be, we can initialize the data
# transforms, image datasets, and the dataloaders. Notice, the models were
# pretrained with the hard-coded normalization values, as described
# `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
#
#
#

# In[ ]:





# In[17]:


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
        self.landmarks_frame = pd.read_csv(os.path.join(root_dir,'F_N_1001.csv'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
#        img = np.zeros([40000, 3, 1])
        rd = self.root_dir.replace('598test','598_processing')
        img_name =         os.path.join(rd,                     self.landmarks_frame.iloc[idx, 0]+'_Surface.ply').replace('\\','/')
        img = np.genfromtxt(img_name, skip_header = 7, skip_footer = 1)
        img = np.concatenate((img, np.zeros([58014-img.shape[0],3])), axis=0)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, D_out)
        sample = {'image': img, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample


# In[18]:


fn = pd.read_csv(os.path.join('D:/seva/598test/train','F_N_1001.csv'))


# In[19]:


len(fn)


# In[20]:


#pwd


# In[21]:


#print(data_dir,os.path.join(data_dir,'/sh_paramters.csv'))


# In[22]:


#print(os.path.isfile('tmean.csv'),os.path.isfile(os.path.join(data_dir,'/sh_paramters.csv')))
#print(not(os.path.isfile('tmean.csv') and os.path.isfile(mainpath+'/sh_paramters.csv')))


# In[23]:


from numpy import linalg as LA


# In[24]:


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
        return {'image': torch.Tensor(image).to(self.device),
                'landmarks': torch.Tensor(landmarks).to(self.device)}
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


# In[25]:


np.zeros(5)+np.min(tmean[2])


# In[26]:


# a=np.zeros([50000,3])
# a = np.pad(a, ((0,229**2-50000),(0,0)), mode='constant')
# a = np.reshape(a, [229,229,3])
# print(a.shape)
# #a.reshape([])


# In[27]:


def getsize(new_hi,ker,srd):
    pad = (0,0)
    dil = np.asarray((1,1))
    return(tuple((np.squeeze((np.asarray(new_hi)+    2*np.asarray(pad)-dil*[np.asarray(ker)-1]+1)/                             np.asarray(srd))).astype(int)))
class TNet(nn.Module):
    def __init__(self) :
        super().__init__()
#         new_h, ampl, m_kernel, m_stride, hidden_dim, C_in, ratio = \
#         tuple(tupa[i] for i in list(range(6, 10))+[11,16,17])
        new_h = 58014
        ampl = 9
        m_kernel = 1
        m_stride = 1
        hidden_dim = 20
        C_in = 1
        new_w = 3
        self.pool = nn.MaxPool2d(m_kernel,m_stride)
        idt = tuple(map(lambda x: x,                 getsize((new_h,new_w,m_kernel,m_stride))))
        input_dim = idt[0]*idt[1]*C_in
#        print(idt)
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, ampl))

    def forward(self, x):
#        print('start',x.shape)
        x = self.pool(x)
#        print('view',x.shape)
        x = x.view(x.shape[0],-1)
        for layer in self.layers[:-1]:
#            print(x.shape)
            x = F.leaky_relu(layer(x))
        out = F.softmax(self.layers[-1](x))
        return out
nnarchitectures = {'TNet':TNet}


# In[28]:


new_h = 58014
new_w = 3
ampl = 9
m_kernel = 1
m_stride = 2
hidden_dim = 20
C_in = 1

#self.pool = nn.MaxPool2d(m_kernel,m_stride)
idt = tuple(map(lambda x: x, getsize((new_h,new_w),m_kernel,m_stride)))
idt = tuple(map(lambda x: x, getsize(idt,2,2)))


# In[29]:


idt


# In[30]:


int((new_h-m_kernel)/m_stride)


# In[31]:


class CNet(torch.nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
#         C_in = 1
#         D_out = 9
        hn1 = int(1e3)
        hn2 = int(1e3)
        hn3 = int(1e3)
        # hn4 = int(1e4)
        # hn5 = int(1e4)
        # hn6 = int(1e4)
        # hn7 = int(1e4)
        self.conv0 = nn.Conv2d(1, 50, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(50, 50, 1)
        self.conv2 = nn.Conv2d(50, 5, 1)
        self.linear1 = torch.nn.Linear(5805, hn1)
        self.linear2 = torch.nn.Linear(hn1, hn2)
        # self.linear3 = torch.nn.Linear(hn2, hn3)
        # self.linear4 = torch.nn.Linear(hn3, hn4)
        # self.linear5 = torch.nn.Linear(hn4, hn5)
#         self.linear6 = torch.nn.Linear(hn5, hn6)
#         self.linear7 = torch.nn.Linear(hn6, hn7)
        self.linear3 = torch.nn.Linear(hn3, 441)

#         self.conv1 = nn.Conv2d(6, 16, 5)

#         self.linear0 = torch.nn.Linear(290000, 441)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
#         print(x.shape)
#         x = self.pool(F.relu(self.conv0(x)))
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
#         print(t.mean(x))
#         x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.shape[0], -1)
#         x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        # print(x.shape)
#     16, 464112
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # x = self.linear4(x)
        # x = self.linear5(x)
        # x = self.linear6(x)
#         x = self.linear7(x)
#         x = self.linear8(x)
#         x = F.relu(self.linear0(x))
#         x = F.relu(self.linear1(x))
        #x = self.linear1(x)
        return x
# class FCL(torch.nn.Module):
#     def __init__(self,ampl):

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         x = self.linear4(x)
#         return x


# ## FCL

# In[32]:


# class FCL(torch.nn.Module):
#     def __init__(self,ampl):
#         super().__init__()
#         hn1 = int(1e4)
#         hn2 = int(1e4)
# #         hn3 = int(1e4)
#         self.linear1 = torch.nn.Linear(1743, hn1)
#         self.linear2 = torch.nn.Linear(hn1, hn2)
#         self.linear3 = torch.nn.Linear(hn2, ampl)
# #         self.linear4 = torch.nn.Linear(hn3, ampl)
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
# #         print(x.shape)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
# #         x = self.linear4(x)
#         return x


# In[33]:


class FCL(torch.nn.Module):
    def __init__(self,ampl):
        super().__init__()
        hn1 = int(1e4)
        hn2 = int(1e4)
        hn3 = int(1e4)
        self.linear1 = torch.nn.Linear(1161, hn1)
        self.linear2 = torch.nn.Linear(hn1, hn2)
        self.linear3 = torch.nn.Linear(hn2, hn3)
        self.linear4 = torch.nn.Linear(hn3, ampl)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
#         print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


# In[34]:


class TNet(nn.Module):
    def __init__(self) :
        super().__init__()
#         self.pool = nn.MaxPool2d(m_kernel,m_stride)
#         idt = tuple(map(lambda x: x, \
#                 getsize((new_h, int(new_h*ratio)),m_kernel,m_stride)))
        input_dim = 1161
        ampl = 441
        rep = '(np.repeat(1024, 10),512)'
        hidar = {rep : (np.repeat(1024, 10),512)}
        hidden_dim = np.hstack(hidar[rep])
#        print(idt)
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, ampl))

    def forward(self, x):
#        print('start',x.shape)
#         x = self.pool(x)
#        print('view',x.shape)
        x = x.view(x.shape[0],-1)
#         print(x.shape)
        for layer in self.layers[:-1]:
#            print(x.shape)
            x = F.leaky_relu(layer(x))
        out = F.softmax(self.layers[-1](x))
        return out
nnarchitectures = {'TNet':TNet}


# In[35]:


ampl = 16
batch_size = 8
data_dir = 'D:/seva/598test'
model_name = "densenet"
ds = 50
data_transforms = {
    'train': transforms.Compose([
#             AmpCrop(ampl),\
#             Minmax(tmean[:,:ampl]),\
            Minmax3Dimage((0.60117054538415,110.972068294924)),\
           Downsample(ds),\
#           Shuffleinput(0),\
#             Normalize(),\
#             Reshape(224),\
            ToTensor(device)
    ]),
    'val': transforms.Compose([
#             AmpCrop(ampl),\
#             Minmax(tmean[:,:ampl]),\
            Minmax3Dimage((0.60117054538415,110.972068294924)),\
           Downsample(ds),\
#           Shuffleinput(0),\
#             Normalize(),\
#             Reshape(224),\
            ToTensor(device)
    ]),
}
print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: Seed3D_Dataset(root_dir=os.path.join(data_dir,x),                                     transform=data_transforms[x])
                  for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset = Seed3D_Dataset(csv_file=mainpath+'/sh_paramters.csv', root_dir=data_dir, transform=data_transform)
# dataloader = DataLoader(dataset, bs,
#                         shuffle=False, num_workers=0)


# In[36]:


# phase = 'train'
# bn = 100
# for i_batch, sample_batched in enumerate(dataloaders_dict[phase]):
#     print(i_batch)
#     if i_batch == bn:
#         break
#     inputs = sample_batched['image']
#     labels = sample_batched['landmarks']


# Create the Optimizer
# --------------------
#
# Now that the model structure is correct, the final step for finetuning
# and feature extracting is to create an optimizer that only updates the
# desired parameters. Recall that after loading the pretrained model, but
# before reshaping, if ``feature_extract=True`` we manually set all of the
# parameter’s ``.requires_grad`` attributes to False. Then the
# reinitialized layer’s parameters have ``.requires_grad=True`` by
# default. So now we know that *all parameters that have
# .requires_grad=True should be optimized.* Next, we make a list of such
# parameters and input this list to the SGD algorithm constructor.
#
# To verify this, check out the printed parameters to learn. When
# finetuning, this list should be long and include all of the model
# parameters. However, when feature extracting this list should be short
# and only include the weights and biases of the reshaped layers.
#
#
#

# In[37]:


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
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
#
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


# Run Training and Validation Step
# --------------------------------
#
# Finally, the last step is to setup the loss for the model, then run the
# training and validation function for the set number of epochs. Notice,
# depending on the number of epochs this step may take a while on a CPU.
# Also, the default learning rate is not optimal for all of the models, so
# to achieve maximum accuracy it would be necessary to tune for each model
# separately.
#
#
#

# In[ ]:





# In[53]:


# Setup the loss fxn
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss(reduction='mean')
bn = 1
num_epochs = 1
# model_name = 'densenet'
lossar = np.zeros([2,num_epochs])
#scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
#for i,lr in enumerate([0.0001, 0.0005, 0.001, 0.005]):
num_classes = 441
ampl = num_classes
# smodel, input_size = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
# Train and evaluate
#smodel = FCL(ampl)
#smodel = TNet()
smodel = CNet()
smodel.to(device)
scratch_optimizer = torch.optim.Adam(smodel.parameters(),             lr=5e-5, betas=(0.9, 0.999),            eps=1e-08, weight_decay=0,             amsgrad=False)
model, lossar = train_model(smodel, dataloaders_dict, criterion, scratch_optimizer, num_epochs, bn, device, is_inception=(model_name=="inception"))


# In[39]:


# x = torch.zeros([8, 1161, 3])
# y = torch.zeros([8, 1161])
# for i in range(x.shape[0]):
#     y[i,:] = torch.sqrt(x[i,:,0]*x[i,:,0]+x[i,:,1]*x[i,:,1]+x[i,:,2]*x[i,:,2])
# y.shape


# In[56]:


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
# ymin, ymax = -, 2
# axes.set_ylim([ymin,ymax])
ymin, ymax = 0,2
axes.set_ylim([ymin,ymax])
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('MSE Loss', fontsize=24)
plt.savefig('D:/seva/598test/plot-output/'+str(int(time.time())))

# In[48]:


# plt.rcParams["figure.figsize"] = (9,5)
# fig = plt.figure()
# axes = plt.gca()
# # ymin, ymax = , 2
# # axes.set_ylim([ymin,ymax])
# labels_text = ['train', 'val']
# for i in range(2):
#     plt.plot(np.arange(lossar.shape[1]),lossar[i,:],label=labels_text[i],linewidth=3)
# axes.set_xticks(np.arange(0, int(num_epochs*1.1)),                          max(int(num_epochs*0.1),1))
# plt.grid()
# plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.,            fontsize = 24)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.autoscale(enable=True, axis='both', tight=None)
# # ymin, ymax = -, 2
# # axes.set_ylim([ymin,ymax])
# ymin, ymax = 0.5,2
# axes.set_ylim([ymin,ymax])
# plt.xlabel('Epoch', fontsize=24)
# plt.ylabel('MSE Loss', fontsize=24)


# ## Return SH coefficient vector from the trained model

# In[42]:


#

# In[57]:


phase = 'val'
bn = 1
ampl = 441
dataloaders = dataloaders_dict
for i_batch, sample_batched in enumerate(dataloaders[phase]):
    if i_batch == bn:
        break
    inputs = sample_batched['image']
    x = inputs
    y = torch.zeros([x.shape[0], x.shape[1]])
    y = y.to(device)
    for i in range(x.shape[0]):
        y[i,:] = torch.sqrt(x[i,:,0]*x[i,:,0]+x[i,:,1]*x[i,:,1]+x[i,:,2]*x[i,:,2])
    y = torch.unsqueeze(y,2)
    gt = sample_batched['landmarks']
    o = model(y)

#    print(model(inputs).shape)
#    output = np.multiply(model(inputs).detach().cpu().numpy(),tmean[3,:ampl]-tmean[2,:ampl])+tmean[2,:ampl]
#    real_output = np.multiply(gt.detach().cpu().numpy(),tmean[3,:ampl]-tmean[2,:ampl])+tmean[2,:ampl]
#     print(output[0])
#     print(real_output[0])


# In[58]:


o = o.detach().cpu().numpy()
gt = gt.detach().cpu().numpy()


# In[61]:


np.savetxt('D:/seva/598test/F_N2output', gt[0], delimiter = ',')


# In[60]:


# o[0]
#
#
# # In[53]:
#
#
# plt.plot(o[0,1:])
#
#
# # In[57]:
#
#
# plt.hist(o[0,1:], bins = 40)
#
#
# # In[90]:
#
#
# def reject_outliers(data, m=2):
#     return data[abs(data - np.mean(data)) < m * np.std(data)]
#
#
# # In[91]:
#
#
# phase = 'train'
# bn = 5000
# ampl = 441
# #dataloaders = dataloaders_dict['val']
# mr = np.zeros([441,150])
# for i_batch, sample_batched in enumerate(dataloaders_dict['val']):
#     if i_batch == bn:
#         break
#     inputs = sample_batched['image']
#     gt = sample_batched['landmarks']
#     o = model(inputs)
#     o = o.detach().cpu().numpy()
#     gt = gt.detach().cpu().numpy()
# #     print(np.mean((o-gt)/gt,axis = 1))
#     for i in range(441):
#         mr[i,i_batch] = np.mean(reject_outliers((o[:,i]-gt[:,i])/gt[:,i], m=2))
# #     if abs(mr[192,i_batch])>100:
# #         print(i_batch)
# #    newar = np.mean(mr,axis=1)
#
# #    print(model(inputs).shape)
# #    output = np.multiply(model(inputs).detach().cpu().numpy(),tmean[3,:ampl]-tmean[2,:ampl])+tmean[2,:ampl]
# #    real_output = np.multiply(gt.detach().cpu().numpy(),tmean[3,:ampl]-tmean[2,:ampl])+tmean[2,:ampl]
# #     print(output[0])
# #     print(real_output[0])
#
#
# # In[87]:
#
#
# o.shape
#
#
# # In[89]:
#
#
# for i_batch, sample_batched in enumerate(dataloaders_dict['val']):
#     if i_batch == 94:
#         inputs = sample_batched['image']
#         gt = sample_batched['landmarks']
#         o = model(inputs)
#         o = o.detach().cpu().numpy()
#         gt = gt.detach().cpu().numpy()
#         print(o[7,192],gt[7,192])
# #         print((o[7,192]-gt[7,192])/gt[7,192])
# #         for i in range(8):
# #             print((o[i,192]-gt[i,192])/gt[i,192])
#
#
# # In[68]:
#
#
# mr.shape
#
#
# # In[92]:
#
#
# plt.plot(np.mean(mr,axis=1))
#
#
# # In[69]:
#
#
# plt.plot(np.mean(mr,axis=1))
#
#
# # In[71]:
#
#
# np.argmin(np.mean(mr,axis=1))
#
#
# # In[64]:
#
#
# newar = np.mean(mr,axis=1)
# newar[192]
#
#
# # In[82]:
#
#
# co = o[5,192]
# cgt = gt[5,192]
# print((co-cgt)/cgt)
#
#
# # In[74]:
#
#
# gt[0,192]
#
#
# # In[55]:
#
#
# plt.plot(mr)
#
#
# # In[43]:
#
#
# bn = 1
# rloss = 0.0
# for i_batch, sample_batched in enumerate(dataloaders_dict['train']):
#     if i_batch == bn:
#         break
#     inputs = sample_batched['image']
#     gt = sample_batched['landmarks']
#     o = model(inputs)
#     loss=criterion(o,gt)
#     rloss+=loss.item()
#
#
# # In[105]:
#
#
# print(rloss/bn)
#
#
# # In[34]:
#
#
# # Setup the loss fxn
# #criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss(reduction='mean')
# bn = 4
# num_epochs = 16
# model_name = 'densenet'
# lrl = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
# lossar = np.zeros([len(lrl),2,num_epochs])
# #scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
# for i,lr in enumerate(lrl):
# #    scratch_model, input_size = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
#     # Train and evaluate
#     smodel = CNet()
#     smodel.to(device)
#     scratch_optimizer = torch.optim.Adam(smodel.parameters(),                 lr, betas=(0.9, 0.999),                eps=1e-08, weight_decay=0,                 amsgrad=False)
#     model, lossar[i] = train_model(smodel, dataloaders_dict, criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
#
#
# # In[197]:
#
#
# lrl = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
# len(lrl)
#
#
# # In[ ]:
#
#
# lossar.shape
#
#
# # In[ ]:
#
#
# print(lossar.shape)
#
#
# # In[42]:
#
#
# for i in range(2):
#     fig = plt.figure()
#     axes = plt.gca()
#     ymin, ymax = -5, 1
#     axes.set_ylim([ymin,ymax])
#     for j in range(lossar.shape[0]):
#         plt.plot(np.arange(lossar.shape[2]),np.log10(lossar[j,i,:]))
#     plt.savefig('plot output/'+                    'num_ampl='+str(ampl)+                    'bn'+str(bn)+                    'bs'+str(batch_size)+                    'epochs'+str(num_epochs)+                    'net_name=Densenet'+                    'lr'+str(0.01)+                    'optimizer adam'+                    'time'+str(time.time())+                '.png', bbox_inches='tight')
#
#
# # In[ ]:
#
#
# linestyles = ['-', '--']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
# fig, axs = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey=True)
# # fig = plt.figure()
# # axes = plt.gca()
# axs = axs.ravel()
# # ymin, ymax = -5, 1
# # axs.set_ylim([ymin,ymax])
# for j in range(lossar.shape[0]):
#     for i in range(2):
#         axs[j].plot(np.arange(lossar.shape[2]),np.log10(lossar[j,i,:]),linestyle=linestyles[i],color=colors[j])
#     plt.savefig('plot output/'+                    'num_ampl='+str(ampl)+                    'bn'+str(bn)+                    'bs'+str(batch_size)+                    'epochs'+str(num_epochs)+                    'net_name=Densenet'+                    'lr'+str(0.01)+                    'optimizer adam'+                    'time'+str(time.time())+                '.png', bbox_inches='tight')
#
#
# # In[ ]:
#
#
# # First create some toy data:
# x = np.linspace(0, 2*np.pi, 400)
# y = np.sin(x**2)
#
# # Create just a figure and only one subplot
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title('Simple plot')
#
# # Create two subplots and unpack the output array immediately
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)
#
# # Create four polar axes and access them through the returned array
# fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
# axs[0, 0].plot(x, y)
# axs[1, 1].scatter(x, y)
#
# # Share a X axis with each column of subplots
# plt.subplots(2, 2, sharex='col')
#
# # Share a Y axis with each row of subplots
# plt.subplots(2, 2, sharey='row')
#
# # Share both X and Y axes with all subplots
# plt.subplots(2, 2, sharex='all', sharey='all')
#
# # Note that this is the same as
# plt.subplots(2, 2, sharex=True, sharey=True)
#
# # Create figure number 10 with a single subplot
# # and clears it if it already exists.
# fig, ax = plt.subplots(num=10, clear=True)
#
#
# # In[ ]:
#
#
# linestyles = ['-', '--']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
# fig = plt.figure()
# axes = plt.gca()
# ymin, ymax = -5, 1
# axes.set_ylim([ymin,ymax])
# for j in range(lossar.shape[0]):
#     for i in range(2):
#         plt.plot(np.arange(lossar.shape[2]),np.log10(lossar[j,i,:]),linestyle=linestyles[i],color=colors[j])
#         plt.savefig('plot output/'+                        'num_ampl='+str(ampl)+                        'bn'+str(bn)+                        'bs'+str(batch_size)+                        'epochs'+str(num_epochs)+                        'net_name=Densenet'+                        'lr'+str(0.01)+                        'optimizer adam'+                        'time'+str(time.time())+                    '.png', bbox_inches='tight')
#
#
# # In[ ]:
#
#
# print(inputs)
#
#
# # ## seed before and after supervised learning
#
# # In[ ]:
#
#
#
#
#
# # Comparison with Model Trained from Scratch
# # ------------------------------------------
# #
# # Just for fun, lets see how the model learns if we do not use transfer
# # learning. The performance of finetuning vs. feature extracting depends
# # largely on the dataset but in general both transfer learning methods
# # produce favorable results in terms of training time and overall accuracy
# # versus a model trained from scratch.
# #
# #
# #
#
# # In[ ]:
#
#
# # Initialize the non-pretrained version of the model used for this run
# model_names = ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]
# scratch_hist = np.zeros([len(model_names),2,num_epochs])
# for i, model_name in  enumerate(model_names):
#     scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
#     scratch_model = scratch_model.to(device)
#     scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
#     scratch_criterion = nn.MSELoss('mean')
#     #scratch_criterion = nn.CrossEntropyLoss()
#     _,scratch_hist[i] = train_model(scratch_model, dataloaders_dict, scratch_criterion,                                  scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
#
#
# # In[ ]:
#
#
# scratch_hist.shape
#
#
# # In[ ]:
#
#
# linestyles = ['-','--']
# for i in range(2):
#     plt.plot(np.arange(num_epochs),scratch_hist[i,:],color = 'tab:blue',linestyle = linestyles[i],linewidth = 3)
# plt.savefig('plot output/'+                'num_ampl='+str(ampl)+                'bn'+str(bn)+                'bs'+str(batch_size)+                'epochs'+str(num_epochs)+                'net_names'+str(net_names)+                'lr'+str(0.001)+                'momentum'+str(0.9)+                'optimizer SGD'+                'scratch'+str(1)+                'time'+str(time.time())+            '.png', bbox_inches='tight')
#
#
# # In[ ]:
#
#
# # Plot the training curves of validation accuracy vs. number
# #  of training epochs for the transfer learning method and
# #  the model trained from scratch
# ohist = []
# shist = []
#
# ohist = [h.cpu().numpy() for h in hist]
# shist = [h.cpu().numpy() for h in scratch_hist]
#
# plt.title("Validation Accuracy vs. Number of Training Epochs")
# plt.xlabel("Training Epochs")
# plt.ylabel("Validation Accuracy")
# plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
# plt.plot(range(1,num_epochs+1),shist,label="Scratch")
# plt.ylim((0,1.))
# plt.xticks(np.arange(1, num_epochs+1, 1.0))
# plt.legend()
# plt.show()
#
#
# # Final Thoughts and Where to Go Next
# # -----------------------------------
# #
# # Try running some of the other models and see how good the accuracy gets.
# # Also, notice that feature extracting takes less time because in the
# # backward pass we do not have to calculate most of the gradients. There
# # are many places to go from here. You could:
# #
# # -  Run this code with a harder dataset and see some more benefits of
# #    transfer learning
# # -  Using the methods described here, use transfer learning to update a
# #    different model, perhaps in a new domain (i.e. NLP, audio, etc.)
# # -  Once you are happy with a model, you can export it as an ONNX model,
# #    or trace it using the hybrid frontend for more speed and optimization
# #    opportunities.
# #
# #
# #
#
# # ## search for the best model
#
# # In[ ]:
#
#
# # Setup the loss fxn
# #criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss(reduction='mean')
# bn = 1
# #model_ft = None
# num_epochs = 10
# net_names = ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]
# #["vgg", "squeezenet"]
# lossa = np.zeros([len(net_names),2,num_epochs])
# #["resnet", "alexnet", "vgg", "squeezenet", "densenet"]
# for i, model_name in enumerate(net_names):
# # Train and evaluate
#     model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
#     print(input_size)
#     model_ft.to(device)
#     model_ft, lossar = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
#     lossa[i] = lossar
#
#
# # In[ ]:
#
#
# print(scratch_hist.shape)
#
#
# # In[ ]:
#
#
# fig = plt.figure()
# axes = plt.gca()
# ymin, ymax = 0, 10
# axes.set_ylim([ymin,ymax])
# linestyles = ['-', '--', '-.', ':', '-', '--',                  '-.', ':', '-', '--', '-.', ':']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# lossa = scratch_hist
# for i in range(lossa.shape[0]):
#     for j in range(lossa.shape[1]):
#         plt.plot(np.arange(num_epochs),                  lossa[i,j,:],                  color = colors[i], linestyle=linestyles[j],                  linewidth=3)
# plt.savefig('plot output/'+                'num_ampl='+str(ampl)+                'bn'+str(bn)+                'bs'+str(batch_size)+                'epochs'+str(num_epochs)+                'net_names'+str(net_names)+                'lr'+str(0.001)+                'momentum'+str(0.9)+                'optimizer SGD'+                'time'+str(time.time())+            '.png', bbox_inches='tight')
#
#
# # In[ ]:
#
#
# linestyles = ['-', '--', '-.', ':', '-', '--',                  '-.', ':', '-', '--', '-.', ':']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# lossa = scratch_hist
# for j in range(lossa.shape[1]):
#     fig = plt.figure()
#     axes = plt.gca()
#     ymin, ymax = 0, 10
#     axes.set_ylim([ymin,ymax])
#     for i in range(lossa.shape[0]):
#         plt.plot(np.arange(num_epochs),                  lossa[i,j,:],                  color = colors[i], linestyle=linestyles[j],                  linewidth=3)
#     plt.savefig('plot output/'+                    'num_ampl='+str(ampl)+                    'bn'+str(bn)+                    'bs'+str(batch_size)+                    'epochs'+str(num_epochs)+                    'net_names'+str(net_names)+                    'lr'+str(0.001)+                    'momentum'+str(0.9)+                    'optimizer SGD'+                    'phase'+str(j)+                    'time'+str(time.time())+                '.png', bbox_inches='tight')
#
#
# # ## search for maximum size of the ply file
#
# # In[ ]:
#
#
# cip = []
# for root, directories, filenames in os.walk(data_dir):
#     for filename in filenames:
#         if filename[-4:] == '.ply':
#             cip.append(os.path.join(root,filename))
#
#
# # In[ ]:
#
#
# len(cip)
# cip[0]
#
#
# # In[ ]:
#
#
# maxsize = 0
# sizear = []
# i = 0
# sumav = 0
# for path in cip:
#     img = np.genfromtxt(path, skip_header = 7, skip_footer = 1)
#     sumav += os.stat(path).st_size/img.shape[0]
#
#     i+=1
#     if i == 50:
#         break
#
# sumav/50
#
# #     sizear.append(img.shape[0])
# #     if img.shape[0]>maxsize:
# #         maxsize = img.shape[0]
# # print(maxsize)
#
#
# # In[ ]:
#
#
# 32.761725985875024
#
#
# # In[ ]:
#
#
# maxsize = 0
# sizear = np.zeros(len(cip))
# i = 0
# sumav = 0
# for i, path in enumerate(cip):
# #    img = np.genfromtxt(path, skip_header = 7, skip_footer = 1)
#     sizear[i] = os.stat(path).st_size/32.761725985875024
#
# #     sizear.append(img.shape[0])
# #     if img.shape[0]>maxsize:
# #         maxsize = img.shape[0]
# # print(maxsize)
# plt.plot(np.sort(sizear))
#
#
# # In[ ]:
#
#
# print(np.max(sizear))
