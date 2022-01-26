# class CcNet(torch.nn.Module):
#     def __init__(self, C_in, D_out):
#         super(CcNet, self).__init__()
#         self.conv0 = nn.Conv2d(C_in, 6, 5)
#         self.conv1 = nn.Conv2d(6, 16, 5)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.linear0 = torch.nn.Linear(32*21*21, 1000)
#         self.linear1 = torch.nn.Linear(1000, D_out)
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv0(x)))
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
#         x = F.relu(self.linear0(x))
#         x = F.relu(self.linear1(x))
#         return x
# class DNet(torch.nn.Module):
#     def __init__(self, C_in, D_out, imsize):
#         super(DNet, self).__init__()
#         self.linear1 = torch.nn.Linear(imsize*imsize*C_in, 100)
#         self.linear2 = torch.nn.Linear(100,100)
#         self.linear3 = torch.nn.Linear(100,100)
#         self.linear4 = torch.nn.Linear(100, D_out)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = F.leaky_relu(self.linear1(x).clamp(min=0))
#         x = F.leaky_relu(self.linear2(x))
#         #print(torch.mean(x))
#         x = F.relu(self.linear3(x))
#         #x = F.relu(self.linear4(x))
#         x = self.linear4(x)
#         return x
class DtNet(torch.nn.Module):
    def __init__(self, C_in, D_out):
        super(DtNet, self).__init__()
        self.linear1 = torch.nn.Linear(200*200*C_in, 100)
        self.linear2 = torch.nn.Linear(100,100)
        self.linear3 = torch.nn.Linear(100,100)
        self.linear4 = torch.nn.Linear(100, D_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.tanh(self.linear1(x).clamp(min=0))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        #x = self.linear4(x)
        return x
class DsNet(torch.nn.Module):
    def __init__(self, C_in, D_out):
        super(DsNet, self).__init__()
        self.linear1 = torch.nn.Linear(200*200*C_in, 10000)
        #torch.nn.init.xavier_uniform(self.linear1.weight)
        self.linear2 = torch.nn.Linear(10000,1000)
        self.linear3 = torch.nn.Linear(1000,100)
        self.linear4 = torch.nn.Linear(100,100)
        self.linear5 = torch.nn.Linear(100,100)
        self.linear6 = torch.nn.Linear(100,100)
        self.linear7 = torch.nn.Linear(100,100)
        self.linear8 = torch.nn.Linear(100,100)
        self.linear9 = torch.nn.Linear(100,100)
        self.linear10 = torch.nn.Linear(100, D_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.linear1(x).clamp(min=0))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        x = F.leaky_relu(self.linear5(x))
        x = F.leaky_relu(self.linear6(x))
        x = F.leaky_relu(self.linear7(x))
        x = F.leaky_relu(self.linear8(x))
        x = F.leaky_relu(self.linear9(x))
        x = self.linear10(x)
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        m.weight.data = torch.nn.init.xavier_uniform(m.weight.data)
                #gain=nn.init.calculate_gain('relu'))
        return x
class triNet(torch.nn.Module):
    def __init__(self, C_in, D_out):
        super(triNet, self).__init__()
        self.linear1 = torch.nn.Linear(200*200*C_in, 1000)
        self.linear2 = torch.nn.Linear(1000,D_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x).clamp(min=0))
        x = F.relu(self.linear2(x))
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        m.weight.data = torch.nn.init.xavier_uniform(m.weight.data)
                #gain=nn.init.calculate_gain('relu'))
        return x
class circlesNet(torch.nn.Module):
    def __init__(self, n):
        super(circlesNet, self).__init__()
        self.linear1 = torch.nn.Linear(200*200*1, 10000)
        #torch.nn.init.xavier_uniform(self.linear1.weight)
        self.linear2 = torch.nn.Linear(10000,1000)
        self.linear3 = torch.nn.Linear(1000,100)
        self.linear4 = torch.nn.Linear(100,100)
        self.linear5 = torch.nn.Linear(100,100)
        self.linear6 = torch.nn.Linear(100,100)
        self.linear7 = torch.nn.Linear(100,100)
        self.linear8 = torch.nn.Linear(100,100)
        self.linear9 = torch.nn.Linear(100,100)
        self.linear10 = torch.nn.Linear(100, 3*n)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x).clamp(min=0))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = F.relu(self.linear7(x))
        x = F.relu(self.linear8(x))
        x = F.relu(self.linear9(x))
        x = F.relu(self.linear10(x))
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        m.weight.data = torch.nn.init.xavier_uniform(m.weight.data)
                #gain=nn.init.calculate_gain('relu'))
        return x
