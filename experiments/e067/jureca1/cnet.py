import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def getsize(hi,ker,srd):
    pad = np.asarray((0,0))
    dil = np.asarray((1,1))
    new_size = np.asarray(hi)
    ker = np.asarray(ker)
    srd = np.asarray(srd)
    return(tuple((np.squeeze(\
        (new_size+2*pad-dil*[ker-1]-1)/srd+1\
        )).astype(int)))
class CNet(torch.nn.Module):
    #network from Henderson, Ferrari article
    def __init__(self, hidden_dim, chidden_dim,
            kernel_sizes, cnum, h, w, usehaf):
            # , lb, ro, C, angles_list,
        super(CNet, self).__init__()
        # (h, w) = (h, w)
        # print('h, w=, ',h, w)
        self.usehaf = usehaf
        if usehaf:
            self.conv0 = nn.Conv2d(cnum, 32, 3, stride=2)
            (h, w) = getsize((h, w),3,2)
            self.conv1 = nn.Conv2d(32, 64, 3)
            (h, w) = getsize((h, w),3,1)
            self.pool = nn.MaxPool2d(2, 2)
            (h, w) = getsize((h, w),2,2)
            self.conv2 = nn.Conv2d(64, 96, 3)
            (h, w) = getsize((h, w),3,1)
            (h, w) = getsize((h, w),2,2)
            self.conv3 = nn.Conv2d(96, 128, 3)
            (h, w) = getsize((h, w),3,1)
            (h, w) = getsize((h, w),2,2)
            self.conv4 = nn.Conv2d(128, 128, 4)
            (h, w) = getsize((h, w),4,1)
            self.bn0 = torch.nn.BatchNorm2d(32)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.bn2 = torch.nn.BatchNorm2d(96)
            self.bn3 = torch.nn.BatchNorm2d(128)
            current_dim = 128*h*w
        else:
#        print(new_h, new_w)
#        self.linear0 = torch.nn.Linear(128*new_h*new_w, 128)
#        self.linear1 = torch.nn.Linear(128, 441)
#        input_dim = 23040
            current_dim = cnum*h*w
        # print(current_dim)
        # self.linear0 = nn.Linear(current_dim, hidden_dim[0])
        # self.linear1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(current_dim, hidden_dim[0]))
        # self.layers.append(nn.Linear(hidden_dim[0], hidden_dim[1]))
        # current_dim = hidden_dim[1]
# #         print(hidden_dim)
        for hdim in hidden_dim:
#             print(hdim)
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
#        self.layers.append(nn.Linear(current_dim, 441))
    def forward(self, x):
        # print('x.shape=',x.shape)
        if self.usehaf:
            x = F.relu(self.bn0(self.conv0(x)))
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = F.relu(self.bn3(self.conv4(x)))
#         print(x.shape)
        x = x.view(x.shape[0], -1)
#         print(x.shape)
#        print(x.shape)
#         x = self.linear0(x)
#         x = self.linear1(x)
#         x = torch.reshape(y,(y.shape[0],int(y.shape[1]/3),3))
#         x = torch.norm(x, dim=2)
        # print('x.shape=', x.shape)
# #         x =
#         y = self.linear0(x)
#         x = self.linear1(y)
#         print('x.shape=',x.shape)
        x = self.layers[0](x)
        # x = self.layers[1](y)
        # if opt.lb == 'pc' and opt.rotate_output:
        #     outputs_1 = torch.reshape(x,
        #     (x.shape[0],3,nsp))
        #     outputs_1 = outputs_1.cuda() if iscuda else outputs_1
        #     outputs_2 = torch.zeros(cbs,3,nsp)
        #     for i in range(cbs):
        #         outputs_2[i,:,:] = torch.matmul(
        #             torch.transpose(torch.squeeze(
        #                 C[int(angles_list[i]/10),:,:]),0,1),
        #                 outputs_1[i,:,:])
        #     outputs_2 = outputs_2.cuda() if iscuda else outputs_2
        #
        # y = None

        # for layer in self.layers[2:]:
        #     x = layer(x)
        return x
