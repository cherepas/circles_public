import torch
class DNet(torch.nn.Module):
    def __init__(self, C_in, D_out, imsize):
        #super(DNet, self).__init__()
        super().__init__()
        self.linear1 = torch.nn.Linear(imsize*imsize*C_in, 100)
        self.linear2 = torch.nn.Linear(100,50)
        #self.linear3 = torch.nn.Linear(100,100)
        self.linear4 = torch.nn.Linear(50, D_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = F.leaky_relu(self.linear1(x).clamp(min=0))
        #print(x.shape)
        x = F.leaky_relu(self.linear2(x))
        #print(x.shape)
        #print(torch.mean(x))
        #x = F.relu(self.linear3(x))
        #x = F.relu(self.linear4(x))
        x = self.linear4(x)
        return x
class CNet(torch.nn.Module):
    def __init__(self, C_in, D_out):
        super(CNet, self).__init__()
        self.conv0 = nn.Conv2d(C_in, 6, 5)
        self.conv1 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear0 = torch.nn.Linear(16*47*47, 1000)
        self.linear1 = torch.nn.Linear(1000, D_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        print(torch.mean(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        #x = self.linear1(x)
        return x
