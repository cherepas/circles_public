import torch
import torch.nn as nn
import torch.nn.functional as F


def getsize(new_hi, ker, srd):
    pad = (0, 0)
    dil = np.asarray((1, 1))
    return(tuple((np.squeeze((np.asarray(new_hi) + 2*np.asarray(pad) -
                  dil*[np.asarray(ker) - 1] + 1) /
                  np.asarray(srd))).astype(int)))

class CNet(torch.nn.Module):
    def __init__(self, hidden_dim,chidden_dim,kernel_sizes,nim, rescale):
        super(CNet, self).__init__()
        current_dim = 58014
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, 441))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        print(x.shape)
        for layer in self.layers[:-1]:
            x = layer(x)
        return x
class CNet(torch.nn.Module):
    #network from Henderson, Ferrari article
    def __init__(self, hidden_dim, chidden_dim, kernel_sizes, nim, rescale_val):
        super(CNet, self).__init__()
        self.conv0 = nn.Conv2d(nim, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 96, 3)
        self.conv3 = nn.Conv2d(96, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 4)
        self.linear = torch.nn.Linear(23040, 441)
        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(96)
        self.bn3 = torch.nn.BatchNorm2d(128)
        input_dim = 23040
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, 441))
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
        for layer in self.layers:
            x = layer(x)
        return x

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
    # 3d-r2n2
    def __init__(self, hidden_dim, chidden_dim, kernel_sizes, nim, rescale_val):
        super(CNet, self).__init__()
        self.chidden_dim = chidden_dim
        srd = 1
        self.pool = nn.MaxPool2d(2, 2)

        self.layers = nn.ModuleList()
        current_dim = nim
        (new_h, new_w) = (rescale_val, 1.8*rescale_val)
        for i in range(len(chidden_dim)):
            self.layers.append(nn.Conv2d(current_dim, chidden_dim[i], kernel_sizes[i]))
            current_dim = chidden_dim[i]
            (new_h, new_w) = getsize((new_h, new_w),kernel_sizes[i],srd)
            (new_h, new_w) = getsize((new_h, new_w),2,2)
            
        current_dim = new_h*new_w*chidden_dim[-1]
#         print('current_dim=',new_h, new_w, nim)
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
#         self.layers.append(nn.Linear(current_dim, 441))
    def forward(self, x):
#         print('input shape=',x.shape)
        for layer in self.layers[:len(self.chidden_dim)]:
#            print(x.shape)
            x = F.relu(self.pool(layer(x)))
#             print('layer=',str(layer),'x shape=',x.shape)
        x = x.view(x.shape[0], -1)
        for layer in self.layers[len(self.chidden_dim):]:
            x = layer(x) 
        return x
    
class CNet(torch.nn.Module):
    #network from Henderson, Ferrari article
    def __init__(self, hidden_dim):
        super(CNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 96, 3)
        self.conv3 = nn.Conv2d(96, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 4)
        self.linear = torch.nn.Linear(23040, 441)
        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(96)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn = torch.nn.BatchNorm1d(441)
        input_dim = 23040
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, 441))
    def forward(self, x):
#         x = torch.unsqueeze(x, 1)
#         x = x.permute(1,0,2,3)
#         print(x.shape)
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv4(x)))
        x = x.view(x.shape[0], -1)
#         print(x.shape)
        x = F.relu(self.bn(self.linear(x)))
        return x
    
class CNet(torch.nn.Module):
    # Custom network with Convolutional and FC layers
    def __init__(self, hidden_dim):
        super(CNet, self).__init__()
#         C_in = 1
#         D_out = 9
        hn1 = int(1e2)
        hn2 = int(1e2)
        hn3 = int(1e2)
        hn4 = int(1e2)
        hn5 = int(1e2)
        hn6 = int(1e2)
        hn7 = int(1e2)
        self.conv0 = nn.Conv2d(1, 25, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(25, 25, 1)
        self.conv2 = nn.Conv2d(25, 5, 1) 
        
        input_dim = 290070
#        print(idt)
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, 441))
        # 290070, 145035, 96690, 72520, 58015
#         self.linear1 = torch.nn.Linear(290070, hn1)
#         self.linear2 = torch.nn.Linear(hn1, hn2)
#         self.linear3 = torch.nn.Linear(hn2, hn3)
#         self.linear4 = torch.nn.Linear(hn3, hn4)
#         self.linear5 = torch.nn.Linear(hn4, hn5)
#         self.linear6 = torch.nn.Linear(hn5, hn6)
#         self.linear7 = torch.nn.Linear(hn6, hn7)
#         self.linear8 = torch.nn.Linear(hn7, 441)

#         self.conv1 = nn.Conv2d(6, 16, 5)

#         self.linear0 = torch.nn.Linear(290000, 441)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
#         print(x.shape)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
#         print(t.mean(x))
#         x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.shape[0], -1)
#         x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
#         print('x.shape after flatting, ', x.shape)
#     16, 464112
        for layer in self.layers[:-1]:
#            print(x.shape)
            x = layer(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         x = self.linear4(x)
#         x = self.linear5(x)
#         x = self.linear6(x)
#         x = self.linear7(x)
#         x = self.linear8(x)
#         x = F.relu(self.linear0(x))
#         x = F.relu(self.linear1(x))
        #x = self.linear1(x)
        return x

class TNet(nn.Module):
    # Network with maxpooling and automatic size calculation 
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


# In[15]:

class FCL(torch.nn.Module):
    # Just FC network
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


# In[17]:


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
