import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self,alpha, scaler=24,actions = 10):
        super(DeepQNetwork, self).__init__()
        self.scalar = scaler
        self.conv1 = nn.Conv2d(1,1*scaler,3,stride=1,padding=1) #input = 6x9x1 output = 6x9x24
        self.conv2 = nn.Conv2d(24, 2*scaler, 3, stride=1, padding=0)  # output = 4x7x72
        self.conv3 = nn.Conv2d(48, 4*scaler, 4, stride=1, padding=0)  #  output =1x4x96= 384
        self.fc1 = nn.Linear(4*4*scaler, 96 )
        self.fc2 = nn.Linear(96, actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr = alpha)
        self.loss = nn.MSELoss()
        self.cuda()

    def forward(self,observation):
        observation = T.Tensor(observation).cuda()
        x = observation.view(-1, 1, 6, 9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x= x.view(-1, 4*4*self.scalar)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

