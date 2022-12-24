import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, cfg, n_cls=10, **kwargs):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, n_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)


def vgg11(**kwargs):
    return VGG(cfg=[64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512], **kwargs)


def vgg13(**kwargs):
    return VGG(cfg=[64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512], **kwargs)


def vgg16(**kwargs):
    return VGG(
        cfg=[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
        **kwargs
    )


def vgg19(**kwargs):
    return VGG(
        cfg=[
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
        ],
        **kwargs
    )
