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
    # 3d-r2n2 copy from gru_net with encoder and decoder, but without LSTM
    def __init__(self, hidden_dim, chidden_dim, kernel_sizes, nim, h, w, usehaf):
        super(CNet, self).__init__()
        n_convfilter = [96, 128, 256, 256, 256, 256, 256]
        n_fc_filters = [1024]
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        (new_h, new_w) = (h, w)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(nim, n_convfilter[0], 7)
        (new_h, new_w) = getsize((new_h, new_w),7,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv2 = nn.Conv2d(n_convfilter[0], n_convfilter[1], 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv3 = nn.Conv2d(n_convfilter[1], n_convfilter[2], 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv4 = nn.Conv2d(n_convfilter[2], n_convfilter[3], 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv5 = nn.Conv2d(n_convfilter[3], n_convfilter[4], 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        self.conv6 = nn.Conv2d(n_convfilter[4], n_convfilter[5], 3)
        (new_h, new_w) = getsize((new_h, new_w),3,1)
        (new_h, new_w) = getsize((new_h, new_w),2,2)
        current_dim =  n_convfilter[5]*new_h*new_w
        self.linear7 = nn.Linear(current_dim, n_fc_filters[0])
        self.unpool = nn.MaxUnpool3d(2, 2)
        self.conv7 = nn.ConvTranspose3d(n_deconvfilter[0], n_deconvfilter[1], 3)
        self.conv8 = nn.ConvTranspose3d(n_deconvfilter[1], n_deconvfilter[2], 3)
        self.conv9 = nn.ConvTranspose3d(n_deconvfilter[2], n_deconvfilter[3], 3)
        self.conv10 = nn.ConvTranspose3d(n_deconvfilter[3], n_deconvfilter[4], 3)
        self.conv11 = nn.ConvTranspose3d(n_deconvfilter[4], n_deconvfilter[5], 3)
    def forward(self, x):
#         x = torch.unsqueeze(x, 1)
#         x = x.permute(1,0,2,3)
#         print(x.shape)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = self.pool(self.conv6(x))
        x = x.view(x.shape[0], -1)
        x = self.linear7(x)
        # sz = torch.pow(x.shape[1], 1/3)
        sz = np.power(x.shape[1],1/3).astype(int)
        print(x.shape)
        x = x[:,:sz**3]
        x = x.reshape(x.shape[0],sz,sz,sz)
        x = self.unpool(x)
        print(x.shape)
        x = nn.LeakyReLU(self.conv7(self.unpool(x)))
        x = nn.LeakyReLU(self.conv8(self.unpool(x)))
        x = nn.LeakyReLU(self.conv9(self.unpool(x)))
        x = nn.LeakyReLU(self.conv10(x))
        x = nn.LeakyReLU(self.conv11(self.unpool(x)))
        x = nn.Softmax(x)
        return x
