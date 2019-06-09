import torch
import torch.nn as nn
import torch.nn.functional as F

class dw_sep_conv(nn.Module):
    def __init__(self, in_feature, out_feature, ker_size, stride):
        super(dw_sep_conv, self).__init__()
        self.padding = ker_size//2
        self.conv1 = nn.Conv2d(in_feature, in_feature,
                               kernel_size=ker_size,
                               stride=stride,
                               padding=self.padding,
                               groups=in_feature)
        self.BatchNorm1 = nn.BatchNorm2d(in_feature)
        self.conv2 = nn.Conv2d(in_feature, out_feature, 1, stride=1)
        self.BatchNorm2 = nn.BatchNorm2d(out_feature)

    def forward(self, input):
        result = self.conv1(input)
        result = F.relu(self.BatchNorm1(result))
        result = self.conv2(result)
        result = F.relu(self.BatchNorm2(result))

        return result
class Convolution(nn.Module):
        def __init__(self,
                     in_feature,
                     out_feature,
                     kernel_size,
                     stride=1,
                     padding=0
                     ):
            super(Convolution, self).__init__()
            self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size, stride=stride, padding=padding)
            self.BatchNorm = nn.BatchNorm2d(out_feature)

        def forward(self, input):
            result = self.conv(input)
            result = F.relu(self.BatchNorm(result))
            return result


class MobileNetv1(nn.Module):
    def __init__(self, class_num=10):
        super(MobileNetv1, self).__init__()
        self.conv1 = Convolution(3, 32, 3, 2, 1)
        self.in_feature = 64
        self.conv2 = dw_sep_conv(32, 64, 3, 1)
        self.conv3 = self.make_layer(2)
        self.conv4 = dw_sep_conv(self.in_feature, self.in_feature*2, 3, 2)
        self.in_feature*=2
        self.conv5 = self.make_layer(5, mode='exfeature')
        self.conv6 = self.make_layer(1)
        self.fc = nn.Linear(1024, class_num)

    def make_layer(self, n, mode='downsample'):
        mode_list = ['downsample', 'exfeature']
        layers = []
        assert mode in mode_list, 'the mode should either be downsample or exfeature'
        if mode=='downsample':
            for i in range(n):
                layers.append(dw_sep_conv(self.in_feature, 2*self.in_feature, 3, 2))
                self.in_feature*=2
                layers.append(dw_sep_conv(self.in_feature, self.in_feature, 3, 1))
        if mode=='exfeature':
            for i in range(n):
                layers.append(dw_sep_conv(self.in_feature, self.in_feature, 3, 1))

        return nn.Sequential(*layers)

    def forward(self, input):
        result = self.conv1(input)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.conv4(result)
        result = self.conv5(result)
        result = self.conv6(result)
        result = F.adaptive_avg_pool2d(result, (1,1))
        result = result.view(result.size(0), -1)
        result = self.fc(result)

        return result




