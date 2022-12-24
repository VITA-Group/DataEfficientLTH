import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import to_adv, to_clean
import torch
from numpy.linalg import norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # # print ratio of norms of residual vs main branch
        # to_print = []
        # for i in range(len(out)):
        #     to_print.append(norm(self.shortcut(x)[i].detach().cpu().numpy()) / norm(out[i].detach().cpu().numpy()))
        # print(sum(to_print)/len(to_print))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Cosine(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(Cosine, self).__init__(in_features, out_features, bias)
        self.s_ = torch.nn.Parameter(torch.zeros(1))

    def loss(self, Z, target):
        s = F.softplus(self.s_).add(1.0)
        l = F.cross_entropy(Z.mul(s), target, weight=None, ignore_index=-100, reduction="mean")
        return l

    def forward(self, input, target):
        logit = F.linear(
            F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), self.bias
        )
        l = self.loss(logit, target)
        return logit, l


class TVMF(Cosine):
    def __init__(self, in_features, out_features, bias=False, kappa=16):
        super(TVMF, self).__init__(in_features, out_features, bias)
        self.register_buffer("kappa", torch.Tensor([kappa]))

    def forward(self, input, target=None):
        assert target is not None
        cosine = F.linear(
            F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), None
        )
        logit = (1.0 + cosine).div(1.0 + (1.0 - cosine).mul(self.kappa)) - 1.0

        if self.bias is not None:
            logit.add_(self.bias)

        l = self.loss(logit, target)
        return logit, l


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, n_cls, pre_conv="full", in_planes=64, in_dim=3, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        if pre_conv == "full":
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_dim, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif pre_conv == "small":
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_dim, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(),
            )
        else:
            raise ValueError("pre_conv should be one of ['full', 'small']")

        self.layer1 = self._make_layer(block, in_planes, n_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, n_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, n_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, n_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 8 * block.expansion, n_cls)

    def _make_layer(self, block, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_(self, x, target=None):
        out = self.pre_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(-1))
        out = out.view(out.size(0), -1)
        if isinstance(self.linear, (Cosine, TVMF)):
            out = self.linear(out, target)
        else:
            out = self.linear(out)
        return out

    def get_fvs(self, x):
        out = self.pre_conv(x)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out1 = F.avg_pool2d(out1, out1.size(-1)).view(out1.size(0), -1)
        out2 = F.avg_pool2d(out2, out2.size(-1)).view(out2.size(0), -1)
        out3 = F.avg_pool2d(out3, out3.size(-1)).view(out3.size(0), -1)
        out4 = F.avg_pool2d(out4, out4.size(-1)).view(out4.size(0), -1)
        return out1, out2, out3, out4

    def forward(self, x, tgt=None, adv_prop=False):
        if adv_prop:
            if not hasattr(self, "attacker"):
                raise ValueError("adv prop training scheme but attacker not defined")
            if self.training:
                self.eval()
                self.apply(to_adv)
                adv_x = self.attacker.attack(x, tgt, self.forward_)
                self.train()
                adv_out = self.forward_(adv_x)
                self.apply(to_clean)
                out = self.forward_(x)
                return out, adv_out
        else:
            return self.forward_(x, tgt)


class ResNetPretrained(nn.Module):
    def __init__(self, net, n_cls, pre_conv="full", in_dim=3, **kwargs):
        super(ResNetPretrained, self).__init__()
        resnet = net(pretrained=True)
        # resnet.load_state_dict(torch.load("simclr_resnet18.ckpt"), strict=False)
        resnet_layers = list(resnet.children())[:-1]

        if pre_conv == "full":
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif pre_conv == "small":
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        else:
            raise ValueError("pre_conv should be one of ['full', 'small']")

        self.layer1 = resnet_layers[4]
        self.layer2 = resnet_layers[5]
        self.layer3 = resnet_layers[6]
        self.layer4 = resnet_layers[7]
        self.linear = nn.Linear(512 * self.layer1[0].expansion, n_cls)

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


PRETRAINED = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def resnet18(**kwargs):
    if kwargs["pretrained"]:
        return ResNetPretrained(PRETRAINED["resnet18"], **kwargs)
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    if kwargs["pretrained"]:
        return ResNetPretrained(PRETRAINED["resnet34"], **kwargs)
    else:
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    if kwargs["pretrained"]:
        return ResNetPretrained(PRETRAINED["resnet50"], **kwargs)
    else:
        return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    if kwargs["pretrained"]:
        return ResNetPretrained(PRETRAINED["resnet101"], **kwargs)
    else:
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    if kwargs["pretrained"]:
        return ResNetPretrained(PRETRAINED["resnet152"], **kwargs)
    else:
        return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    net = resnet18(n_cls=10, pre_conv="small", in_planes=12)
    print(f"model params: {sum(p.numel() for p in net.parameters())/1e6}M")
