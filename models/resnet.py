"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.
Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

from utils import load_state_dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import autocast

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# This class is from LDAM: https://github.com/kaidic/LDAM-DRW.
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, dropout=None, num_classes=1000, use_norm=False, use_block=True, reduce_dimension=False,
                 layer3_output_dim=None, layer4_output_dim=None, load_pretrained_weights=False, returns_feat=False,
                 s=30):
        self.inplanes = 64
        self.use_block, self.use_norm = use_block, use_norm
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        self.layer4 = self._make_layer(block, layer4_output_dim, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        if self.use_block:
            self.rb_block = self._make_layer(block, 256, layers[3], stride=2)
            self.rb_block = self._make_layer(block, layer4_output_dim, layers[3], stride=2)
        else:
            self.rb_block = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.use_norm:
            self.linear = NormedLinear(layer4_output_dim * block.expansion, num_classes)
            self.linear_rb = NormedLinear(layer4_output_dim * block.expansion, num_classes)
        else:
            s = 1
            self.linear = nn.Linear(layer4_output_dim * block.expansion, num_classes)
            self.linear_rb = nn.Linear(layer4_output_dim * block.expansion, num_classes)

        self.returns_feat = returns_feat
        self.s = s

        if load_pretrained_weights:
            caffe_model = True
            if caffe_model:
                print('Loading Caffe Pretrained ResNet 152 Weights.')
                pretrained_weights_state_dict = torch.load('./data/caffe_resnet152.pth')
            else:
                print('Loading Places-LT Pretrained ResNet 152 Weights.')
                pretrained_weights_state_dict = torch.load('./data/places_lt_pretrained.pth')['state_dict_best'][
                    'feat_model']
                pretrained_weights_state_dict = {k[7:]: v for k, v in
                                                 pretrained_weights_state_dict.items()}  # remove "module."
            should_ignore = lambda param_name: param_name.startswith('fc')  # It's called fc in caffe model.
            should_copy = lambda param_name: param_name.startswith('layer4')

            for k in list(pretrained_weights_state_dict.keys()):
                if should_ignore(k):
                    pretrained_weights_state_dict.pop(k)
                    print("Ignored when loading the model:", k)
                if should_copy(k):
                    pretrained_weights_state_dict[k.replace('layer4', 'rb_block')] = pretrained_weights_state_dict[k]

            # The number of parameters may mismatch since we don't have num_batches_tracked in the caffe model.
            load_state_dict(self, pretrained_weights_state_dict, no_ignore=True)

            # print("Warning: We allow training on all layers.")
            print("Warning: We allow training on layer 3 and layer 4.")
            # print("Warning: We allow training on layer 4 and classifier.")

            # should_train = lambda param_name: param_name.startswith('layer1') or param_name.startswith('layer2') or param_name.startswith('layer3') or param_name.startswith(
            #     'layer4') or param_name.startswith('rb_block') or param_name.startswith('linear')
            should_train = lambda param_name: param_name.startswith('layer3') or param_name.startswith('layer4') or param_name.startswith('rb_block') or param_name.startswith('linear')
            # should_train = lambda param_name: param_name.startswith('layer4') or param_name.startswith('rb_block') or param_name.startswith('linear')
            for name, param in self.named_parameters():
                if not should_train(name):
                    param.requires_grad_(False)
                else:
                    print("Allow gradient on:", name)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            if "feature_cb" in kwargs:
                x = self.layer4(x)
                x = self.avgpool(x)
                return x
            elif "feature_rb" in kwargs:
                x = self.rb_block(x)
                x = self.avgpool(x)
                return x
            if "classifier_cb" in kwargs:
                if self.layer4:
                    x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                if self.use_dropout:
                    x = self.dropout(x)
                x = self.linear(x)
                return x * self.s

            elif "classifier_rb" in kwargs:
                if self.rb_block:
                    x = self.rb_block(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                if self.use_dropout:
                    x = self.dropout(x)
                x = self.linear_rb(x)
                return x * self.s

            if self.layer4:
                x_1 = self.layer4(x)
                x_1 = self.avgpool(x_1)
            else:
                x_1 = self.avgpool(x)
            x_1 = x_1.view(x_1.size(0), -1)
            if self.use_dropout:
                x_1 = self.dropout(x_1)
            x_1 = self.linear(x_1)
            if "use_experts" in kwargs:
                if self.rb_block:
                    x_2 = self.rb_block(x)
                    x_2 = self.avgpool(x_2)
                else:
                    x_2 = self.avgpool(x)
                x_2 = x_2.view(x_2.size(0), -1)
                if self.use_dropout:
                    x_2 = self.dropout(x_2)
                x_2 = self.linear_rb(x_2)
                return x_1, x_2
            else:
                return x_1

def resnet10(num_classes, use_norm=True, use_block=True):
    return ResNet(BasicBlock, [1, 1, 1, 1], dropout=None,
                  num_classes=num_classes, use_norm=use_norm, use_block=use_block)

def resnet50(num_classes, use_norm=True, use_block=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], dropout=None,
                  num_classes=num_classes, use_norm=use_norm, use_block=use_block)

def resnet152(num_classes, use_norm=True, use_block=True, load_pretrained_weights=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], dropout=None,
                  num_classes=num_classes, use_norm=use_norm, use_block=use_block, load_pretrained_weights=load_pretrained_weights)
