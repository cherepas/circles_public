#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch as t
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
device = t.device("cuda" if t.cuda.is_available() else "cpu")
C_in = 3 #number of views of the one object that are inserted in the color channel
D_out = 441 #number of featers of spherical harmonics to regress
epochs = 1 #how many epochs
lr = 5e-5 #learning rate
bn = 1 #how many batches
bs = 1 #how many images in a batch
modelname = 'CNet'
lossb = np.zeros([2, 2, epochs])
sc = 100
nh = 5*sc #image has ratio 5:9.
nw = 9*sc
t.manual_seed(0)
def getbatch(bcr):
    label = t.rand([bs, D_out]).to(device)
    data = t.rand([bs, C_in, nh, nw]).to(device)
    return(data,label)
def getsize(szi,ker,srd):
    pad = (0,0)
    dil = np.asarray((0,0))
    return(tuple((np.squeeze((szi+2*np.asarray(pad)-dil*\
    [np.asarray(ker)-1]+1)/np.asarray(srd))).astype(int)))
class CNet(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(C_in, 8, (5,5))
        self.conv1 = nn.Conv2d(8, 16, (5,5))
        self.pool = nn.MaxPool2d(2,2)
        sz = getsize((nh,nw),(5,5),(1,1))
        sz = getsize(sz,(2,2),(2,2))
        sz = getsize(sz,(5,5),(1,1))
        sz = getsize(sz,2,2)
        sz = np.asarray(sz)-4-int((sc/2)%2)
        self.linear0 = t.nn.Linear(16*sz[0]*sz[1], 1000)
        self.linear1 = t.nn.Linear(1000, D_out)
    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        return x
nnarchitectures = {'CNet' : CNet}
def multit():
    model = nnarchitectures[modelname]()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999), \
    eps=1e-08, weight_decay=0, amsgrad=False)
    lossa = np.zeros([2, epochs])
    xval, xtest = [t.zeros([bs, C_in, nh, nw]).to(device)]*2
    yval, ytest = [t.zeros([bs, D_out]).to(device)]*2
    [xtest, ytest] = getbatch(0)
    [xval, yval] = getbatch(1)
    for j in range(epochs):
        for i in range(bn):
            [xtrain, ytrain] = getbatch(i+2)
            try:
                del loss
            except:
                1
            criterion = t.nn.MSELoss(reduction='mean')
            xtrain = xtrain.detach()
            #xtrain contains images of the circle
            y_pred = model(xtrain)
            #y_pred consists of coordinates of the circle center and its radius
            # Forward pass: Compute predicted y by passing x to the model
            #calculating loss between predicted parameters and parameters
            #which were used for generation
            loss = criterion(y_pred, ytrain)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lossa[0][j] = loss.item()
        lossa[1][j] = criterion(model(xval), yval).item()
        del loss
    return(model, lossa)
def traplot():
    ymin, ymax = -4, -2
    plt.rcParams["figure.figsize"] = (18,10)
    linestyles = ['-', '--', '-.', ':']
    colors = ['red', 'green']
    axes = plt.gca()
    labels_text = [['train', 'validation'], ['train rand input', \
    'validation rand input']]
    axes.set_ylim([ymin,ymax])
    for trt in range(2):
        for isr in range(2):
            plt.plot(np.arange(lossb.shape[2]), \
            np.log(lossb[isr][trt][:]), label=labels_text[trt][isr], \
            color = colors[isr], linestyle=linestyles[trt], linewidth=3)
    axes.set_xticks(np.arange(0, int(lossb[0][0].shape[0]*1.1), \
    int(lossb[0][0].shape[0]*0.1)))
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', \
    borderaxespad=0., fontsize = 24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('log(MSE Loss)', fontsize=24)
    plt.savefig('size='+str([nh, nw])+'bn'+str(bn)+'bs'+str(bs)+'epochs'+\
    str(epochs)+'layers'+str(layers)+'lr'+str(lr)+'etime'+\
    format(timelapse,".2f")+'noisein'+str(randinit)+'.png', bbox_inches='tight')
    plt.show()
_, lossb[0,:,:], timelapse = multit()
