import torch.nn as nn
import torch.nn.functional as F
import torch

class LENET(nn.Module):
    def __init__(self):
        super(LENET,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=16, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)


    def forward(self, input):
        batch_size = input.size(0)
        output = F.relu(self.conv1(input))
        output = F.max_pool2d(output,kernel_size=2, stride=2)
        output = F.relu(self.conv2(output))
        output = F.max_pool2d(output, kernel_size=2, stride=2)
        output = output.view(batch_size, -1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output