import torch
import torch.nn as nn
import torch.nn.functional as F
import math



eps = 1e-12


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def squash(s, dim=-1):
    mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
    mag = torch.sqrt(mag_sq)
    v = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return v



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
        def __init__(self, block, num_blocks, channels):
            super(ResNet, self).__init__()
            self.in_planes = 128
            planes = 128
            
            self.conv1 = nn.Conv2d(channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.layer1 = self._make_layer(block, planes, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 2*planes, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 4*planes, num_blocks[2], stride=2)
            
            self.apply(weights_init)
            
        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)
            
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            return out
    
def resnet20(cfg_data, num_caps, caps_size, depth):
    return ResNet(BasicBlock, [3, 3, 3], cfg_data['channels'])

def resnet32(cfg_data, num_caps, caps_size, depth):
    return ResNet(BasicBlock, [5, 5, 5], cfg_data['channels'])

def resnet44(cfg_data, num_caps, caps_size, depth):
    return ResNet(BasicBlock, [7, 7, 7], cfg_data['channels'])

def resnet56(cfg_data, num_caps, caps_size, depth):
    return ResNet(BasicBlock, [9, 9, 9], cfg_data['channels'])

def resnet110(cfg_data, num_caps, caps_size, depth):
    return ResNet(BasicBlock, [18, 18, 18], cfg_data['channels'])
