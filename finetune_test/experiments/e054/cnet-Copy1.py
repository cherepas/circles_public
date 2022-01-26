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
    def __init__(self, hidden_dim, chidden_dim, kernel_sizes, nim, h, w, usehaf):
        super(CNet, self).__init__()
        (new_h, new_w) = (h, w)
        self.usehaf = usehaf
        if usehaf:
            self.conv0 = nn.Conv2d(nim, 32, 3, stride=1)
            (new_h, new_w) = getsize((new_h, new_w),3,1)
            self.conv1 = nn.Conv2d(32, 32, 3, stride=1)
            (new_h, new_w) = getsize((new_h, new_w),3,1)
#             print(new_h,new_w)

            self.conv2 = nn.Conv2d(32, 64, 3, stride=2)            
            (new_h, new_w) = getsize((new_h, new_w),3,2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)
            self.conv4 = nn.Conv2d(64, 64, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)

            self.conv5 = nn.Conv2d(64, 128, 3, stride=2)            
            (new_h, new_w) = getsize((new_h, new_w),3,2)
            self.conv6 = nn.Conv2d(128, 128, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)
            self.conv7 = nn.Conv2d(128, 128, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)

            self.conv8 = nn.Conv2d(128, 256, 3, stride=2)            
            (new_h, new_w) = getsize((new_h, new_w),3,2)
            self.conv9 = nn.Conv2d(256, 256, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)
            self.conv10 = nn.Conv2d(256, 256, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)

            self.conv11 = nn.Conv2d(256, 512, 3, stride=2)            
            (new_h, new_w) = getsize((new_h, new_w),3,2)
            self.conv12 = nn.Conv2d(512, 512, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)
            self.conv13 = nn.Conv2d(512, 512, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)

            self.conv14 = nn.Conv2d(512, 512, 3, stride=1)            
            (new_h, new_w) = getsize((new_h, new_w),3,1)
            self.conv15 = nn.Conv2d(512, 512, 5, stride=2)            
            (new_h, new_w) = getsize((new_h, new_w),5,2)

            current_dim = 512*new_h*new_w
        else:
#        print(new_h, new_w)
#        self.linear0 = torch.nn.Linear(128*new_h*new_w, 128)
#        self.linear1 = torch.nn.Linear(128, 441)
#        input_dim = 23040
            current_dim = nim*new_h*new_w
        self.layers = nn.ModuleList()
        self.linear0 = nn.Linear(current_dim, 128)
        current_dim = 128
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
#        self.layers.append(nn.Linear(current_dim, 441))
    def forward(self, x):
        if self.usehaf:
            x = F.relu(self.conv0(x))
#             print(x.shape)
            x = F.relu(self.conv1(x))
#             print(x.shape)
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = F.relu(self.conv7(x))
            x = F.relu(self.conv8(x))
            x = F.relu(self.conv9(x))
            x = F.relu(self.conv10(x))
            x = F.relu(self.conv11(x))
            x = F.relu(self.conv12(x))
            x = F.relu(self.conv13(x))
            x = F.relu(self.conv14(x))
            x = F.relu(self.conv15(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear0(x))
        for layer in self.layers:
            x = layer(x)
        return x