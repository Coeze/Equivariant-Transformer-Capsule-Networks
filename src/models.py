import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.coordinates import logpolar_grid
from src.networks import BasicCNN, EquivariantPosePredictor, TransformerLayer
from src.transformers import RotationScale, ScaleX, TransformerSequence, Translation

import src.resnet as r
from src.train import DATASET_CONFIGS

import math

class SelfRouting2d(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=3, stride=1, padding=1, pose_out=False):
        super(SelfRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.pose_out = pose_out

        if pose_out:
            self.W1 = nn.Parameter(torch.FloatTensor(self.kkA, B*D, C))
            nn.init.kaiming_uniform_(self.W1.data)

        self.W2 = nn.Parameter(torch.FloatTensor(self.kkA, B, C))
        self.b2 = nn.Parameter(torch.FloatTensor(1, 1, self.kkA, B))

        nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.b2.data, 0)

    def forward(self, a, pose):
        # a: [b, A, h, w]
        # pose: [b, AC, h, w]
        b, _, h, w = a.shape

        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        if hasattr(self, 'W1'):
            # [b, l, kkA, BD]
            pose_out = torch.matmul(self.W1, pose).squeeze(-1)
            # [b, l, kkA, B, D]
            pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B]
        logit = torch.matmul(self.W2, pose).squeeze(-1) + self.b2

        # [b, l, kkA, B]
        r = torch.softmax(logit, dim=3)

        # [b, kkA, l]
        a = F.unfold(a, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a = a.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a = a.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA, 1]
        a = a.view(b, l, self.kkA, 1)

        # [b, l, kkA, B]
        ar = a * r
        # [b, l, 1, B]
        ar_sum = ar.sum(dim=2, keepdim=True)
        # [b, l, kkA, B, 1]
        coeff = (ar / (ar_sum)).unsqueeze(-1)

        # [b, l, B]
        # a_out = ar_sum.squeeze(2)
        a_out = ar_sum / a.sum(dim=2, keepdim=True)
        a_out = a_out.squeeze(2)

        # [b, B, l]
        a_out = a_out.transpose(1,2)

        if hasattr(self, 'W1'):
            # [b, l, B, D]
            pose_out = (coeff * pose_out).sum(dim=2)
            # [b, l, BD]
            pose_out = pose_out.view(b, l, -1)
            # [b, BD, l]
            pose_out = pose_out.transpose(1,2)

        oh = ow = math.floor(l**(1/2))

        a_out = a_out.view(b, -1, oh, ow)
        if hasattr(self, 'W1'):
            pose_out = pose_out.view(b, -1, oh, ow)
        else:
            pose_out = None

        return a_out, pose_out
    
class ETCAPS(nn.Module):
    def __init__(self, args):
        super(ETCAPS, self).__init__()
        self.cfg_data = DATASET_CONFIGS[args.dataset]
        channels, classes = self.cfg_data['channels'], self.cfg_data['classes']

        self.num_caps = args.num_caps
        self.caps_size = args.caps_size
        planes = 64
        self.depth = args.depth
        equivariant_transformers = TransformerSequence(
            Translation(predictor_cls=EquivariantPosePredictor, in_channels=channels, nf=32),
            RotationScale(predictor_cls=EquivariantPosePredictor, in_channels=channels, nf=32),
            ScaleX(predictor_cls=EquivariantPosePredictor, in_channels=channels, nf=32)
        ) 
        self.et_layer = TransformerLayer(
            transformer=equivariant_transformers,
            coords=logpolar_grid, 
        )
        
        self.backbone = r.__dict__[args.encoder](self.cfg_data, args.num_caps, args.caps_size, args.depth)


        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for d in range(1, args.depth):
            stride = 2 if d == 1 else 1
            self.conv_layers.append(SelfRouting2d(args.num_caps, args.num_caps, args.caps_size, args.caps_size, kernel_size=3, stride=stride, padding=1, pose_out=True))
            self.norm_layers.append(nn.BatchNorm2d(args.caps_size*args.num_caps))

        final_shape = 8 if args.depth == 1 else 4

        self.conv_a = nn.Conv2d(4*planes, args.num_caps, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(4*planes, args.num_caps*args.caps_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(args.num_caps)
        self.bn_pose = nn.BatchNorm2d(args.num_caps*args.caps_size)
        self.fc = SelfRouting2d(args.num_caps, classes, args.caps_size, 1, kernel_size=final_shape, padding=0, pose_out=False)

        self.apply(r.weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.et_layer(x)
        out = self.backbone(x)
        a, pose = self.conv_a(out), self.conv_pose(out)
        a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

        for m, bn in zip(self.conv_layers, self.norm_layers):
            a, pose = m(a, pose)
            pose = bn(pose)

        a, _ = self.fc(a, pose)
        out = a.view(a.size(0), -1)
        out = out.log()

        return out

    def forward_activations(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        a = torch.sigmoid(self.bn_a(self.conv_a(out)))

        return a
    
    
# Just a normal self-routing capsule network without equivariant transformers
class SRCAPS(nn.Module):
    def __init__(self, args):
        super(ETCAPS, self).__init__()
        self.cfg_data = DATASET_CONFIGS[args.dataset]
        channels, classes = self.cfg_data['channels'], self.cfg_data['classes']

        self.num_caps = args.num_caps
        self.caps_size = args.caps_size
        planes = 64
        self.depth = args.depth
        self.backbone = r.__dict__[args.encoder](self.cfg_data, args.num_caps, args.caps_size, args.depth)


        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for d in range(1, args.depth):
            stride = 2 if d == 1 else 1
            self.conv_layers.append(SelfRouting2d(args.num_caps, args.num_caps, args.caps_size, args.caps_size, kernel_size=3, stride=stride, padding=1, pose_out=True))
            self.norm_layers.append(nn.BatchNorm2d(args.caps_size*args.num_caps))

        final_shape = 8 if args.depth == 1 else 4

        self.conv_a = nn.Conv2d(4*planes, args.num_caps, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(4*planes, args.num_caps*args.caps_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(args.num_caps)
        self.bn_pose = nn.BatchNorm2d(args.num_caps*args.caps_size)
        self.fc = SelfRouting2d(args.num_caps, classes, args.caps_size, 1, kernel_size=final_shape, padding=0, pose_out=False)

        self.apply(r.weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.et_layer(x)
        out = self.backbone(x)
        a, pose = self.conv_a(out), self.conv_pose(out)
        a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

        for m, bn in zip(self.conv_layers, self.norm_layers):
            a, pose = m(a, pose)
            pose = bn(pose)

        a, _ = self.fc(a, pose)
        out = a.view(a.size(0), -1)
        out = out.log()

        return out

    def forward_activations(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        a = torch.sigmoid(self.bn_a(self.conv_a(out)))

        return a