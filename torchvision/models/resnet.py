import torch
import torch.nn as nn
from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class TridentBlock(torch.nn.Module):
    expansion = 4
    def __init__(self, input_channels, output_channels, stride = 1, padding = [1,2,3], dilation = [1,2,3], 
                 downsample = None, last = False, norm_layer=None, fast_mode = False):
        super(TridentBlock, self).__init__()  
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.shared_weights4convolution1 = torch.nn.Parameter(torch.randn(output_channels, input_channels , 1, 1))
        self.shared_weights4convolution2 = torch.nn.Parameter(torch.randn(output_channels, output_channels, 3, 3))
        self.shared_weights4convolution3 = torch.nn.Parameter(torch.randn(output_channels*self.expansion, output_channels, 1,1))       
        self.bn11 = norm_layer(output_channels)
        self.bn12 = norm_layer(output_channels)
        self.bn13 = norm_layer(output_channels*self.expansion)        
        self.bn21 = norm_layer(output_channels)
        self.bn22 = norm_layer(output_channels)
        self.bn23 = norm_layer(output_channels*self.expansion)        
        self.bn31 = norm_layer(output_channels)
        self.bn32 = norm_layer(output_channels)
        self.bn33 = norm_layer(output_channels*self.expansion)        
        self.relu1 = torch.nn.ReLU(inplace = True)
        self.relu2 = torch.nn.ReLU(inplace = True)
        self.relu3 = torch.nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding
        self.dilation = dilation 
        self.last = last
        self.fast_mode = fast_mode

#----====----====----====----====----====----====----====----====----====----====----====----====----====
    def forward_branch_1(self,x):
        residual = x
        # conv 1x1
        output = torch.nn.functional.conv2d(x, self.shared_weights4convolution1, bias = None)
        output = self.bn11(output)
        output = self.relu1(output)
        # conv 3x3
        output = torch.nn.functional.conv2d(output, self.shared_weights4convolution2, bias = None,
                                            stride = self.stride, padding = self.padding[0], 
                                            dilation = self.dilation[0])
        output = self.bn12(output)
        output = self.relu1(output)
        # conv 1x1
        output = torch.nn.functional.conv2d(output, self.shared_weights4convolution3, bias = None)
        output = self.bn13(output)
        if self.downsample is not None:
            residual = self.downsample(x)        
        output += residual
        output = self.relu1(output)
        return output
    
#----====----====----====----====----====----====----====----====----====----====----====----====----====
    def forward_branch_2(self,x): # bran
        residual = x
        # conv 1x1
        output = torch.nn.functional.conv2d(x, self.shared_weights4convolution1, bias = None)
        output = self.bn21(output)
        output = self.relu2(output)
        # conv 3x3
        output = torch.nn.functional.conv2d(output, self.shared_weights4convolution2, bias = None,
                                            stride = self.stride, padding = self.padding[1], 
                                            dilation = self.dilation[1])
        output = self.bn22(output)
        output = self.relu2(output)
        # conv 1x1
        output = torch.nn.functional.conv2d(output, self.shared_weights4convolution3, bias = None)
        output = self.bn23(output)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        output += residual
        output = self.relu2(output)
        return output
    
#----====----====----====----====----====----====----====----====----====----====----====----====----====
    def forward_branch_3(self,x): # bran
        residual = x
        # conv 1x1
        output = torch.nn.functional.conv2d(x, self.shared_weights4convolution1, bias = None)
        output = self.bn31(output)
        output = self.relu3(output)
        # conv 3x3
        output = torch.nn.functional.conv2d(output, self.shared_weights4convolution2, bias = None,
                                            stride = self.stride, padding = self.padding[2], 
                                            dilation = self.dilation[2])
        output = self.bn32(output)
        output = self.relu3(output)
        # conv 1x1
        output = torch.nn.functional.conv2d(output, self.shared_weights4convolution3, bias = None)
        output = self.bn33(output)
        if self.downsample is not None:
            residual = self.downsample(x)
        output += residual
        output = self.relu3(output)
        return output
    
#----====----====----====----====----====----====----====----====----====----====----====----====----====
#----====----====----ВЫХОЫ ИЗ ТРИDЕНТ СЛОЯ---====----====----====----====----====----====----====----====
#----====----====----====----====----====----====----====----====----====----====----====----====----====
    def forward(self,x):
        feature_list = list()
        # if self.downsample is not None:
        if isinstance(x,type(list())):
            feature_list.append(self.forward_branch_1(x[0]))
            feature_list.append(self.forward_branch_2(x[1]))
            feature_list.append(self.forward_branch_3(x[2])) 
        else:
            feature_list.append(self.forward_branch_1(x))
            feature_list.append(self.forward_branch_2(x))
            feature_list.append(self.forward_branch_3(x))
        if self.last:
            if self.fast_mode:
                feature_list = feature_list[1]
            else:
                """check this immediately!!!!!"""
                feature_list = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim = 1) 
        return feature_list
#----====----====----====----====----====----====----====----====----====----====----====----====----====
#----====----====----====----====----====----====----====----====----====----====----====----====----====
#----====----====----====----====----====----====----====----====----====----====----====----====----====
#----====----====----====----====----====----====----====----====----====----====----====----====----====

class ResNet(nn.Module):

    def __init__(self, block, trident_block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, fast_mode = False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.fast_mode = fast_mode
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if trident_block is None:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        else:
            self.layer4 = self._make_layer1(trident_block, 512, layers[3], stride=2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _make_layer1(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
            torch.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            self._norm_layer(planes * block.expansion),
      )
        layers = []
        layers.append(block(self.inplanes, planes, stride = stride, downsample = downsample, norm_layer = norm_layer,
                            fast_mode = self.fast_mode, ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i != blocks -1 :
                layers.append(block(self.inplanes, planes, norm_layer = norm_layer))
            else:
                layers.append(block(self.inplanes, planes, norm_layer = norm_layer, last = True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, trident_block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, trident_block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def trident_resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('trident_resnet50', Bottleneck, TridentBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def trident_resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('trident_resnet101', Bottleneck, TridentBlock, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def trident_resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('trident_resnet152', Bottleneck, TridentBlock, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

