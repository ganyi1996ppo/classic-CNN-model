import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleBlock(nn.Module):
    def __init__(self, in_feature, out_feature, stride, g, downsample):
        super(ShuffleBlock, self).__init__()
        self.g = g
        self.downsample = downsample
        self.stride = stride
        self.conv1 = nn.Conv2d(in_feature, out_feature//4, 1)
        self.dwconv = nn.Conv2d(out_feature//4,
                                out_feature//4,
                                3,
                                stride,
                                padding=1,
                                groups=out_feature//4)
        self.BatchNorm1 = nn.BatchNorm2d(out_feature//4)
        self.BatchNorm2 = nn.BatchNorm2d(out_feature//4)
        self.BatchNorm3 = nn.BatchNorm2d(out_feature)
        self.conv3 = nn.Conv2d(out_feature//4, out_feature, 1)

    def ChannelShuffle(self, feature_map):
        B,C,H,W = feature_map.size()
        feature_map = feature_map.view(B, self.g,C//self.g,H,W)
        feature_map = feature_map.permute(0,2,1,3,4).contiguous().view(B,C,H,W)
        return feature_map

    def forward(self, input):
        result = self.conv1(input)
        result = F.relu(self.BatchNorm1(result))
        result = self.ChannelShuffle(result)
        result = self.dwconv(result)
        result = self.BatchNorm2(result)
        result = self.conv3(result)
        result = self.BatchNorm3(result)
        shortcut = self.downsample(input)
        result = torch.cat((result, shortcut), dim=1) if self.stride==2 else result+shortcut
        result = F.relu(result)
        return result

class ShuffleNet(nn.Module):
    def __init__(self, out_feature, block, block_num, class_num, g):
        super(ShuffleNet, self).__init__()
        self.in_feature = 24
        self.g = g
        self.out_feature = out_feature
        self.block = block
        self.conv1 = nn.Conv2d(3, 24, 3 ,2, padding=1)
        self.stage2 = self.make_layer(block_num[0])
        self.stage3 = self.make_layer(block_num[1])
        self.stage4 = self.make_layer(block_num[2])
        self.fc = nn.Linear(self.out_feature, class_num)
        self.init_weight()

    def make_layer(self, block_num):
        layers = []
        if isinstance(block_num, int):
            block1 = block_num
            block2 = block_num
        else:
            block1 = block_num[0]
            block2 = block_num[1]
        for i in range(block1):
            downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            layers.append(ShuffleBlock(self.in_feature,
                                       self.out_feature-self.in_feature,
                                       stride=2,
                                       g=self.g,
                                       downsample=downsample))
        for i in range(block2):
            downsample = nn.Sequential()
            layers.append(ShuffleBlock(self.out_feature,
                                       self.out_feature,
                                       stride=1,
                                       g=self.g,
                                       downsample=downsample))
        self.in_feature = self.out_feature
        self.out_feature*=2

        return nn.Sequential(*layers)

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

    def forward(self, input):
        result = self.conv1(input)
        result = F.max_pool2d(result, kernel_size=3, stride=2, padding=1)
        result = self.stage2(result)
        result = self.stage3(result)
        result = self.stage4(result)
        result = F.adaptive_avg_pool2d(result, (1,1))
        result = result.view(result.size(0), -1)
        result = self.fc(result)

        return result

def ShuffleNetG1():
    return ShuffleNet(144, ShuffleBlock, [(1,3),(1,7),(1,3)], 10, 1)

def ShuffleNetG2():
    return ShuffleNet(200, ShuffleBlock, [(1,3),(1,7),(1,3)], 10, 2)

def ShuffleNetG3():
    return ShuffleNet(240, ShuffleBlock, [(1,3),(1,7),(1,3)], 10, 3)

#def ShuffleNetG4():
#    return ShuffleNet(272, ShuffleBlock, [(1,3),(1,7),(1,3)], 10, 4)


#def ShuffleNetG8():
#    return ShuffleNet(384, ShuffleBlock, [(1,3),(1,7),(1,3)], 10, 8)





