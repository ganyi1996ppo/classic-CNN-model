import torch
import torch.nn as nn
import torch.nn.functional as F

class DarkNet_Block(nn.Module):
    expansion = 2
    def __init__(self, in_feature, plane):
        super(DarkNet_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, plane, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(plane, self.expansion*plane, kernel_size=3, stride=1, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(plane)
        self.BatchNorm2 = nn.BatchNorm2d(self.expansion*plane)
        if self.expansion*plane !=in_feature:
            self.shortcut = nn.Sequential(nn.Conv2d(in_feature, self.expansion*plane, kernel_size=1),
                                          nn.BatchNorm2d(self.expansion*plane))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input):
        conv1 = self.conv1(input)
        bn1 = F.leaky_relu(self.BatchNorm1(conv1), negative_slope=0.1)
        conv2 = self.conv2(bn1)
        bn2 = F.leaky_relu(self.BatchNorm2(conv2), negative_slope=0.1)
        result = F.leaky_relu(self.shortcut(input)+bn2)
        return result

class DarkNet(nn.Module):
    def __init__(self, Blocktype, Blocks):
        super(DarkNet, self).__init__()
        self.Blocktype = Blocktype
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.in_feature = 32
        self.conv3 = self.make_layers(Blocks[0])
        self.conv4 = nn.Conv2d(self.in_feature, self.in_feature*2, 3, stride=2, padding=1)
        self.conv5 = self.make_layers(Blocks[1])
        self.conv6 = nn.Conv2d(self.in_feature, self.in_feature*2, 3, stride=2, padding=1)
        self.conv7 = self.make_layers(Blocks[2])
        self.conv8 = nn.Conv2d(self.in_feature, self.in_feature*2, 3, stride=2, padding=1)
        self.conv9 = self.make_layers(Blocks[3])
        self.conv10 = nn.Conv2d(self.in_feature, self.in_feature*2, 3, stride=2, padding=1)
        self.conv11 = self.make_layers(Blocks[4])
        self.fc = nn.Linear(self.in_feature, 1000)


    def make_layers(self, Block):
        layers = []
        for i in range(Block):
            self.Blocktype(self.in_feature*2, self.in_feature)
        self.in_feature*=2
        return nn.Sequential(*layers)

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        conv11 = self.conv11(conv10)
        Bs = conv11.size(0)
        avg = F.adaptive_avg_pool2d(conv11, (1,1)).view(Bs, -1)
        result = self.fc(avg)

        return result

def DarkNet53():
    return DarkNet(DarkNet_Block, [1,2,8,8,4])

def test():
    img = torch.randn(4, 3, 256, 256)
    model = DarkNet53()
    print(model(img))

#test()



