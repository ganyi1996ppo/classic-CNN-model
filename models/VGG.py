import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {'vgg11':[1,1,2,2,2],
       'vgg13':[2,2,2,2,2],
       'vgg16':[2,2,(2,1),(2,1),(2,1)],
       'vgg19':[2,2,4,4,4]
       }

class VGG(nn.Module):
    def __init__(self, M):
        super(VGG,self).__init__()
        self.base_out = 64
        self.feature_stop = 3
        self.vgg_conv = self.make_conv_layers(cfg[M])
        self.fc1 = nn.Linear(512, 10)
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

    def make_conv_layers(self, config):
        vgg = nn.ModuleList()
        in_feature = 3
        out_feature = self.base_out
        for i, num_layer in enumerate(config):
            num_3 = 0
            num_1 = 0
            if isinstance(num_layer, tuple):
                num_3 = num_layer[0]
                num_1 = num_layer[1]
            else:
                num_3 = num_layer
            for j in range(num_3):
                vgg.append(nn.Conv2d(in_feature, out_feature, 3, padding=1))
                vgg.append(nn.BatchNorm2d(out_feature))
                vgg.append(nn.ReLU())
                in_feature = out_feature
            for j in range(num_1):
                vgg.append(nn.Conv2d(in_feature, out_feature, 1, padding=1))
                vgg.append(nn.BatchNorm2d(out_feature))
                vgg.append(nn.ReLU())
            vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))
            out_feature=out_feature*2 if i<self.feature_stop else out_feature
        vgg.append(nn.AdaptiveMaxPool2d((1,1)))
        return vgg

    def forward(self, input):
        result = input
        for layer in self.vgg_conv:
            result = layer(result)
        result = result.view(-1, 512)
        result = self.fc1(result)
        return result


