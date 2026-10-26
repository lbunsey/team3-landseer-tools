"""Built-in model definitions that add_model can write into a repo that lacks them.

Every template follows the conventions backdoor-defense repos usually rely on:
  - the factory takes (num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs), so it can be
    called like the repo's own factories;
  - every BatchNorm layer is built with norm_layer(num_features), so defenses that swap in
    their own BatchNorm subclass (e.g. EP/BNP) see every layer;
  - each convolution is registered right before the BatchNorm that normalizes it;
  - global average pooling, so the model works for 32x32 (CIFAR) and 64x64 (Tiny-ImageNet) inputs.
"""

import re

HEADER = '"""{title}, added by the Landseer add_model tool."""\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'

VGG = '''
CFG = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
              512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10, norm_layer=nn.BatchNorm2d):
        super(VGG, self).__init__()
        layers, in_planes = [], 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_planes, v, kernel_size=3, padding=1, bias=False),
                           norm_layer(v), nn.ReLU()]
                in_planes = v
        self.features = nn.Sequential(*layers)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def vgg11(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    return VGG(CFG['vgg11'], num_classes, norm_layer)


def vgg13(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    return VGG(CFG['vgg13'], num_classes, norm_layer)


def vgg16(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    return VGG(CFG['vgg16'], num_classes, norm_layer)


def vgg19(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    return VGG(CFG['vgg19'], num_classes, norm_layer)
'''

PREACT = '''
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(PreActBlock, self).__init__()
        self.bn1 = norm_layer(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, norm_layer=nn.BatchNorm2d):
        super(PreActResNet, self).__init__()
        self._norm_layer = norm_layer
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.bn = norm_layer(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for stride in [stride] + [1] * (num_blocks - 1):
            layers.append(PreActBlock(self.in_planes, planes, stride, self._norm_layer))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def preactresnet18(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    return PreActResNet([2, 2, 2, 2], num_classes, norm_layer)


def preactresnet34(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    return PreActResNet([3, 4, 6, 3], num_classes, norm_layer)
'''

MOBILENET = '''
class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride, norm_layer):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                norm_layer(out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride); first strides are 1 for small images
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10, norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        layers, in_planes = [], 32
        for expansion, out_planes, num_blocks, stride in self.cfg:
            for stride in [stride] + [1] * (num_blocks - 1):
                layers.append(Block(in_planes, out_planes, expansion, stride, norm_layer))
                in_planes = out_planes
        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = norm_layer(1280)
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def mobilenetv2(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    return MobileNetV2(num_classes, norm_layer)
'''

PREACT_NOTE = ("PreAct ResNet puts each BatchNorm before its convolution. Defenses that pair every "
               "BatchNorm with the convolution just before it (e.g. CLP) may need adapting.")

# file stem -> (title, source, factory names, note)
FAMILIES = {
    "vgg": ("VGG with BatchNorm for small images", VGG, ["vgg11", "vgg13", "vgg16", "vgg19"], ""),
    "preact_resnet": ("Pre-activation ResNet for small images", PREACT,
                      ["preactresnet18", "preactresnet34"], PREACT_NOTE),
    "mobilenetv2": ("MobileNetV2 for small images", MOBILENET, ["mobilenetv2"], ""),
}


def find_template(model_name):
    """'VGG-16', 'vgg16_bn' and 'PreAct-ResNet18' map to a factory; returns
    (factory, file stem, file source, note), or None if no template fits."""
    key = re.sub(r"[^a-z0-9]", "", model_name.lower())
    key = re.sub(r"^(vgg\d+)bn$", r"\1", key)
    for stem, (title, body, factories, note) in FAMILIES.items():
        if key in factories:
            return key, stem, HEADER.format(title=title) + body.lstrip("\n"), note
    return None


def template_names():
    return [f for _, _, factories, _ in FAMILIES.values() for f in factories]
