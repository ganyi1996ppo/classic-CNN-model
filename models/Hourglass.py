import torch
import torch.nn as nn
import torch.nn.functional as F

class convolution(nn.Module):
    def __init__(self,
                 in_feature,
                 out_feature,
                 kernel_size,
                 stride=1,
                 padding=0,
                 norm=True):
        super(convolution,self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_feature) if norm else nn.Sequential()

    def forward(self, input):
        conv = self.conv(input)
        bn = F.relu(self.bn(conv))
        return bn

class fully_connected(nn.Module):
    def __init__(self,
                 in_feature,
                 out_feature,
                 bias=False,
                 norm=True):
        super(fully_connected, self).__init__()
        self.fc = nn.Linear(in_feature, out_feature, bias)
        self.bn = nn.BatchNorm1d(out_feature) if norm else nn.Sequential()

    def forward(self, input):
        fc = self.fc(input)
        bn = F.relu(self.bn(fc))
        return bn

class residual_basic(nn.Module):
    expansion = 1
    def __init__(self,
                 in_feature,
                 plane,
                 stride=1
                 ):
        super(residual_basic, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, plane, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(plane)
        self.conv2 = nn.Conv2d(plane, plane*self.expansion, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(plane*self.expansion)
        if stride!=1 or plane*self.expansion!=in_feature:
            self.shortcut = nn.Sequential(nn.Conv2d(in_feature, plane*self.expansion, kernel_size=3, stride=stride, padding=1),
                                          nn.BatchNorm2d(plane*self.expansion))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input):
        conv1 = self.conv1(input)
        bn1 = F.relu(self.bn1(conv1))
        conv2 = self.conv2(bn1)
        bn2 = F.relu(self.bn2(conv2))
        shortcut = self.shortcut(input)
        result = F.relu(bn2+shortcut)

        return result

def make_layers(modules, layer, in_feature, out_feature, **kwargs):
        layers = [layer(in_feature, out_feature, **kwargs)]
        for i in range(modules-1):
            layers.append(layer(out_feature, out_feature, **kwargs))
        return nn.Sequential(*layers)

def make_layers_reverse(modules, layer, in_feature, out_feature, **kwargs):
    layers = []
    for i in range(modules-1):
        layers.append(layer(in_feature, in_feature, **kwargs))
    layers = [layer(out_feature, in_feature, **kwargs)]
    return nn.Sequential(*layers)

def upsample(dim):
    return nn.Upsample(scale_factor=2)

def downsample(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)


class Hourglass(nn.Module):
    def __init__(self, dims, modules, layer = residual_basic,central_layer=make_layers,
                 make_layer=make_layers, make_layer_reverse=make_layers_reverse,
                 shortcut=make_layers, upsample=upsample, downsample=downsample,**kwargs):
        super(Hourglass, self).__init__()
        assert len(dims)==len(modules), 'the module number should be equal to the dimension number. '
        cur_dim = dims[0]
        nex_dim = dims[1]
        cur_module = modules[0]
        nex_module = modules[1]
        self.shortcut = shortcut(cur_module, layer, cur_dim, cur_dim, **kwargs)
        self.down = downsample(cur_dim)
        self.conv1 = make_layer(cur_module, layer, cur_dim, nex_dim, **kwargs)
        self.conv2 = Hourglass(dims[1:],modules[1:], layer,central_layer,
                               make_layer, make_layer_reverse,
                               shortcut, upsample, downsample, **kwargs) if len(dims)>2 else \
            central_layer(nex_module, layer, nex_dim, nex_dim, **kwargs)
        self.conv3 = make_layer_reverse(cur_module, layer, cur_dim, nex_dim, **kwargs)
        self.up = upsample(cur_dim)

    def forward(self, input):
        shortcut = self.shortcut(input)
        down = self.down(input)
        conv1 = self.conv1(down)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        up = self.up(conv3)
        result = up+shortcut
        return result

def test():
    hourglass = Hourglass([3,256, 256, 512], [2,2,2,2]).cuda()
    img = torch.randn(1, 3, 512, 512).cuda()
    result = hourglass(img)
    print(result)

test()






