#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')


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

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.special import sph_harm
import sys
import shutil

# In[4]:
from cnet import *


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#shutil.copyfile('./cnet.py', dst)
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
#parser.add_argument('-ampcrop', type=bool, default=False)
parser.add_argument('-minmax', dest='minmax', action='store_true')
parser.add_argument('-no-minmax', dest='minmax', action='store_false')
parser.set_defaults(minmax=False)
parser.add_argument('-minmax3dimage', dest='minmax3dimage', action='store_true')
parser.add_argument('-no-minmax3dimage', dest='minmax3dimage', action='store_false')
parser.set_defaults(minmax3dimage=False)
parser.add_argument('-normalize', dest='normalize', action='store_true')
parser.add_argument('-no-normalize', dest='normalize', action='store_false')
parser.set_defaults(normalize=False)
parser.add_argument('-center', dest='center', action='store_true')
parser.add_argument('-no-center', dest='center', action='store_false')
parser.set_defaults(center=False)
#parser.add_argument('-minmax', type=bool, default=False)
#parser.add_argument('-minmax3dimage', type=bool, default=False)
parser.add_argument('-downsample', type=int, default=1)
#parser.add_argument('-normalize', type=bool, default=False)
#parser.add_argument('-center', type=bool, default=False)
parser.add_argument('-datatype', type=int, default=1001)
parser.add_argument('-ampl', type=int, default=441)
parser.add_argument('-use_adasum',type=bool, default = False)
parser.add_argument('-gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('-expnum',type=str, default = '111')
parser.add_argument('-hidden_dim',type=str, default = 'hidden_dim = np.hstack((np.repeat(128, 7),441))')
opt = parser.parse_args()
print('epoch='+str(opt.epoch)+'\n'\
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
                'datatype='+str(opt.datatype)+'\n'\
                'ampl='+str(opt.ampl)+'\n'\
     )


rank = hvd.rank()
if rank == 0:
    dir1 = '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/598test/plot_output/'+opt.expnum
    dirname = '/p/home/jusers/cherepashkin1/jureca/cherepashkin1/598test/plot_output/'+opt.expnum+'/'+str(int(time.time()))+'/'
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

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
                'datatype='+str(opt.datatype)+'\n'\
                'ampl='+str(opt.ampl)+'\n'\
#                'time_elapsed='+str(time_elapsed)
              ) 
    file.close() 
    




# In[ ]:




