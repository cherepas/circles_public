import torch
import torch.nn as nn
import torch.nn.functional as F
class CNet(torch.nn.Module):
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
#         print('x shape before FC layer',x.shape)
#         x = F.relu(self.bn(self.linear(x)))
        for layer in self.layers[:-1]:
#            print(x.shape)
            x = layer(x)
        return x

