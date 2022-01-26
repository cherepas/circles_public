import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNet(torch.nn.Module):
    # 3d-r2n2
    def __init__(self, hidden_dim, nim):
        super(CNet, self).__init__()
        n_convfilter = [96, 128, 256, 256, 256, 256, 256]
#         n_fc_filters = [1024]
#         n_deconvfilter = [128, 128, 128, 64, 32, 2]
        
        self.conv0 = nn.Conv2d(nim, n_convfilter[0], 7)
        self.conv1 = nn.Conv2d(n_convfilter[0], n_convfilter[1], 3)
        self.conv2 = nn.Conv2d(n_convfilter[1], n_convfilter[2], 3)
        self.conv3 = nn.Conv2d(n_convfilter[2], n_convfilter[3], 3)
        self.conv4 = nn.Conv2d(n_convfilter[3], n_convfilter[4], 3)
        self.conv5 = nn.Conv2d(n_convfilter[4], n_convfilter[5], 3)
#        self.conv6 = nn.Conv2d(n_convfilter[5], n_convfilter[6], 3)

        self.pool = nn.MaxPool2d(2, 2)
#        self.linear = torch.nn.Linear(23040, 441)
        input_dim = 1024
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for cdim in chidden_dim:
            
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, 441))
    def forward(self, x):
#         x = torch.unsqueeze(x, 1)
#         x = x.permute(1,0,2,3)
#         print(x.shape)
        x = F.relu(self.pool(self.conv0(x)))
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.pool(self.conv5(x)))
#        x = F.relu(self.pool(self.conv6(x)))
        x = x.view(x.shape[0], -1)
#         print('x shape before FC layer',x.shape)
#         x = F.relu(self.bn(self.linear(x)))
        for layer in self.layers[:-1]:
#            print(x.shape)
            x = layer(x)
        return x

