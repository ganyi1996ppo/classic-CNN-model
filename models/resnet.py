import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, in_feature, plane,  stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, plane, kernel_size=3, stride=stride)
        self.conv2 = nn.Conv2d(plane, plane*self.expansion, kernel_size=3, stride= stride)
        self.BatchNorm1 = nn.BatchNorm2d(plane)
        self.BatchNorm2 = nn.BatchNorm2d(plane)
        if stride!=1 or in_feature!=self.expansion*plane:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_feature, self.expansion*plane, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*plane)
            )
        else:
            self.short_cut = nn.Sequential()

    def forward(self, input):
        result = input
        result = self.conv1(result)
        result = F.relu(self.BatchNorm1(result))
        result = self.conv2(result)
        result = self.BatchNorm2(result)
        identity = self.short_cut(input)
        result += identity
        result = F.relu(result)

        return result

class BottleNeck(nn.Module):
    expansion=4
    def __init__(self, in_feature, plane, stride=1):
        super(BottleNeck, self).__init__()
        self.in_feature = in_feature
        self.outfeature = plane
        self.stride = stride
        self.conv1 = nn.Conv2d(in_feature, plane, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, padding=1, stride=self.stride, bias=False)
        self.conv3 = nn.Conv2d(plane, self.expansion * plane, kernel_size=1, stride=1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(plane)
        self.BatchNorm2 = nn.BatchNorm2d(plane)
        self.BatchNorm3 = nn.BatchNorm2d(self.expansion * plane)
        if stride!=1 or in_feature!=self.expansion*plane:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_feature, plane * self.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(plane * self.expansion)
            )
        else:
            self.short_cut = nn.Sequential()

    def forward(self, input):
        result = self.conv1(input)
        result = F.relu(self.BatchNorm1(result))
        result = self.conv2(result)
        result = F.relu(self.BatchNorm2(result))
        result = self.conv3(result)
        result = self.BatchNorm3(result)
        identity = self.short_cut(input)
        result += identity
        result = F.relu(result)

        return result


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, stride, num_class=100):
        super(ResNet, self).__init__()
        self.in_feature = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(64)
        self.conv2 = self.make_layer(block, num_blocks[0], 64, stride[0])
        self.conv3 = self.make_layer(block, num_blocks[1], 128, stride[1])
        self.conv4 = self.make_layer(block, num_blocks[2], 256, stride[2])
        self.conv5 = self.make_layer(block, num_blocks[3], 512, stride[3])
        self.fc = nn.Linear(512*block.expansion, num_class)
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

    def make_layer(self, block, num_blcok, plane, stride):
        strides = [stride]+[1]*(num_blcok-1)
        layer = []
        for stride in strides:
            layer.append(block(self.in_feature, plane, stride))
            self.in_feature = plane * block.expansion

        return nn.Sequential(*layer)

    def forward(self, input):
        result = self.conv1(input)
        result = self.BatchNorm1(result)
        result = F.relu(result)
        result = F.max_pool2d(result, kernel_size=3, stride=2, padding=1)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.conv4(result)
        result = self.conv5(result)
        result = F.adaptive_avg_pool2d(result, (1,1))
        result = result.view(result.size(0), -1)
        result = self.fc(result)
        return result

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2],[1,2,2,2])
def resnet34():
    return ResNet(BasicBlock, [3,4,6,3],[1,2,2,2])
def resnet50():
    return ResNet(BottleNeck, [3,4,6,3],[1,2,2,2])
def resnet101():
    return ResNet(BottleNeck, [3,4,23,3],[1,2,2,2])
def resnet152():
    return ResNet(BottleNeck, [3,8,36,3], [1,2,2,2])
