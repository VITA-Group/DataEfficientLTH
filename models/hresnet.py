import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np


def dct_filters(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k**2 - int(not DC)
    else:
        if level <= k:
            nf = level * (level + 1) // 2 - int(not DC)
        else:
            r = 2 * k - 1 - level
            nf = k**2 - r * (r + 1) // 2 - int(not DC)
    filter_bank = np.zeros((nf, k, k), dtype=np.float32)
    m = 0
    for i in range(k):
        for j in range(k):
            if (not DC and i == 0 and j == 0) or (not level is None and i + j >= level):
                continue
            for x in range(k):
                for y in range(k):
                    filter_bank[m, x, y] = math.cos((math.pi * (x + 0.5) * i) / k) * math.cos(
                        (math.pi * (y + 0.5) * j) / k
                    )
            if l1_norm:
                filter_bank[m, :, :] /= np.sum(np.abs(filter_bank[m, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank[m, :, :] *= (2.0 / k) * ai * aj
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups, 1, 1, 1))
    return torch.FloatTensor(filter_bank)


class Harm2d(nn.Module):
    def __init__(
        self,
        ni,
        no,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        dilation=1,
        use_bn=False,
        level=None,
        DC=True,
        groups=1,
    ):
        super(Harm2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dct = nn.Parameter(
            dct_filters(
                k=kernel_size,
                groups=ni if use_bn else 1,
                expand_dim=1 if use_bn else 0,
                level=level,
                DC=DC,
            ),
            requires_grad=False,
        )

        nf = self.dct.shape[0] // ni if use_bn else self.dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm2d(ni * nf, affine=False)
            self.weight = nn.Parameter(
                nn.init.kaiming_normal_(
                    torch.Tensor(no, ni // self.groups * nf, 1, 1),
                    mode="fan_out",
                    nonlinearity="relu",
                )
            )
        else:
            self.weight = nn.Parameter(
                nn.init.kaiming_normal_(
                    torch.Tensor(no, ni // self.groups, nf, 1, 1),
                    mode="fan_out",
                    nonlinearity="relu",
                )
            )
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None

    def forward(self, x):
        if not hasattr(self, "bn"):
            filt = torch.sum(self.weight * self.dct, dim=2)
            x = F.conv2d(
                x,
                filt,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            return x
        else:
            x = F.conv2d(
                x,
                self.dct,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=x.size(1),
            )
            x = self.bn(x)
            x = F.conv2d(x, self.weight, bias=self.bias, padding=0, groups=self.groups)
            return x


def harm3x3(in_planes, out_planes, stride=1, level=None):
    return Harm2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        use_bn=False,
        level=level,
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, harm=True, level=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if harm:
            self.harm1 = harm3x3(inplanes, planes, stride, level=level)
            self.harm2 = harm3x3(planes, planes, level=level)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.harm1(x) if hasattr(self, "harm1") else self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.harm2(out) if hasattr(self, "harm2") else self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, harm=True, level=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if harm:
            self.harm2 = harm3x3(planes, planes, stride, level=level)
        else:
            self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.harm2(out) if hasattr(self, "harm2") else self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        n_blocks,
        n_cls,
        pre_conv="full",
        in_planes=64,
        harm_root=True,
        harm_res_blocks=True,
        levels=[None, None, None, None],
        in_dim=3,
        **kwargs
    ):
        super(ResNet, self).__init__()
        self.inplanes = in_planes

        if pre_conv == "full":
            if harm_root:
                conv = Harm2d(
                    in_dim,
                    self.inplanes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                    use_bn=True,
                )
            else:
                conv = nn.Conv2d(
                    in_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
                )
            self.pre_conv = nn.Sequential(
                conv,
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif pre_conv == "small":
            if harm_root:
                conv = Harm2d(
                    in_dim,
                    self.inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    use_bn=True,
                )
            else:
                conv = nn.Conv2d(
                    in_dim, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
                )
            self.pre_conv = nn.Sequential(
                conv,
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(),
            )
        else:
            raise ValueError("pre_conv should be one of ['full', 'small']")

        self.layer1 = self._make_layer(
            block, 64, n_blocks[0], harm=harm_res_blocks, level=levels[0]
        )
        self.layer2 = self._make_layer(
            block, 128, n_blocks[1], stride=2, harm=harm_res_blocks, level=levels[1]
        )
        self.layer3 = self._make_layer(
            block, 256, n_blocks[2], stride=2, harm=harm_res_blocks, level=levels[2]
        )
        self.layer4 = self._make_layer(
            block, 512, n_blocks[3], stride=2, harm=harm_res_blocks, level=levels[3]
        )
        self.linear = nn.Linear(512 * block.expansion, n_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, harm=True, level=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, harm, level))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, harm=harm, level=level))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
