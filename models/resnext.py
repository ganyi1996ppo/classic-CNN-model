import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupBlock(nn.Module):
    expansion = 2
    def __init__(self, in_feature, plane, cardinality, stride):
        super(GroupBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, plane, kernel_size=1, stride=1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(plane)
        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, padding=1, groups=cardinality, stride=stride, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(plane)
        self.conv3 = nn.Conv2d(plane, self.expansion*plane, kernel_size=1, stride=1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(self.expansion*plane)
        self.shortcut = nn.Sequential()
        if stride!=1 or self.expansion*plane!=in_feature:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_feature, self.expansion*plane, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*plane)
            )

    def forward(self, input):
        result = self.conv1(input)
        result = F.relu(self.BatchNorm1(result))
        result = self.conv2(result)
        result = F.relu(self.BatchNorm2(result))
        result = self.conv3(result)
        result = self.BatchNorm3(result)
        shortcut = self.shortcut(input)
        result = F.relu(result+shortcut)

        return result

class ResNext(nn.Module):
    def __init__(self, block, num_block, strides, cardinality, num_class=10):
        super(ResNext, self).__init__()
        self.in_feature = 64
        self.cardinality = cardinality
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(self.in_feature)
        self.conv2 = self.make_resnext_layer(128, block, num_block[0], stride=strides[0])
        self.conv3 = self.make_resnext_layer(256, block, num_block[1], stride=strides[1])
        self.conv4 = self.make_resnext_layer(512, block, num_block[2], stride=strides[2])
        self.conv5 = self.make_resnext_layer(1024, block, num_block[3], stride=strides[3])
        self.fc = nn.Linear(self.in_feature, num_class)
        self.init_weight()

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def make_resnext_layer(self, plane, block, block_number, stride):
        strides = [stride]+[1]*(block_number-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_feature, plane, self.cardinality, stride))
            self.in_feature = plane*block.expansion

        return nn.Sequential(*layers)

    def forward(self, input):
        result = self.conv1(input)
        result = self.BatchNorm1(result)
        result = F.relu(result)
        result = F.max_pool2d(result, kernel_size=3, padding=1, stride=2)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.conv4(result)
        result = self.conv5(result)
        result = F.adaptive_avg_pool2d(result, (1,1))
        result = result.view(result.size(0), -1)
        result = self.fc(result)

        return result

def ResNext50():
    return ResNext(GroupBlock, [3,4,6,3], [1,2,2,2], 32)

def ResNext101():
    return ResNext(GroupBlock, [3,4,23,3], [1,2,2,2], 32)

def ResNext152():
    return ResNext(GroupBlock, [3,8,36,3], [1,2,2,2], 32)
