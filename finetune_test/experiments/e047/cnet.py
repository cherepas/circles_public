import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def getsize(new_hi,ker,srd): 
    pad = np.asarray((0,0))
    dil = np.asarray((1,1))
    new_size = np.asarray(new_hi)
    ker = np.asarray(ker)
    srd = np.asarray(srd)
    return(tuple((np.squeeze(\
        (new_size+2*pad-dil*[ker-1]-1)/srd+1\
        )).astype(int)))
class CNet(torch.nn.Module):
    #network from Henderson, Ferrari article
    def __init__(self, hidden_dim, chidden_dim, kernel_sizes, nim, h, w):
        super(CNet, self).__init__()
        (new_h, new_w) = (h, w)
        self.conv0 = nn.Conv2d(nim, 32, 3, stride=2)
        (new_h, new_w) = getsize((new_h, new_w),3,2)
        self.conv1 = nn.Conv2d(32, 64, 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        self.pool = nn.MaxPool2d(2, 2)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv2 = nn.Conv2d(64, 96, 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv3 = nn.Conv2d(96, 128, 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv4 = nn.Conv2d(128, 128, 4)
        (new_h, new_w) = getsize((new_h, new_w),4,1)
#        print(new_h, new_w)
#        self.linear0 = torch.nn.Linear(128*new_h*new_w, 128)
#        self.linear1 = torch.nn.Linear(128, 441)
        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(96)
        self.bn3 = torch.nn.BatchNorm2d(128)
#        input_dim = 23040
        current_dim = 128*new_h*new_w
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
#        self.layers.append(nn.Linear(current_dim, 441))
    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv4(x)))
        x = x.view(x.shape[0], -1)
#        print(x.shape)
#        x = self.linear0(x)
#        x = self.linear1(x)
        for layer in self.layers:
            x = layer(x)
        return x