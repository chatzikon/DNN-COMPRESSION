'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, cfg0,cfg1, stride):
        super(Block, self).__init__()
        self.stride = stride

        #planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, cfg0, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg0)
        self.conv2 = nn.Conv2d(cfg0, cfg1, kernel_size=3, stride=stride, padding=1, groups=cfg0, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg1)
        self.conv3 = nn.Conv2d(cfg1, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes,  stride)
    # cfg = [(1,  16,  1),
    #        (6,  24,  1),  # NOTE: change stride 2 -> 1 for CIFAR10
    #        (6, 24, 1),
    #        (6,  32,  1),
    #        (6, 32, 1),
    #        (6, 32,  1),
    #        (6,  64, 2),
    #        (6, 64,  1),
    #        (6, 64,  1),
    #        (6, 64,  1),
    #        (6,  96,  1),
    #        (6, 96,  1),
    #        (6, 96,  1),
    #        (6, 96,  1),
    #        (6, 160,  2),
    #        (6, 160,  1),
    #        (6, 160,  1),
    #        (6, 320,  1)]
    cfg= [32,32,96,96,144,144,144,144,192,192,192,192,(192,2),192,384,384,384,384,384,384,384,384,576,576,576,576,576,576,576,576,(576,2),576,960,960,960,960, 960,960 ]
    cfg_res=[32,16,24,24,32,32,32,64,64,64,64,96,96,96,96,96,160,160,160,320]

    def __init__(self, num_classes=10,cfg=None):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(cfg)
        self.conv2 = nn.Conv2d(self.cfg_res[-1], 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, cfg=None):
        layers = []




        if cfg!=None:
            for i in range(len(cfg)):
                if isinstance(self.cfg[i], int):
                    self.cfg[i]=cfg[i]
                else:
                    temp=list(self.cfg[i])
                    temp[0]=cfg[i]
                    self.cfg[i] = tuple(temp)
        else:
            self.cfg=[32,32,96,96,144,144,144,144,192,192,192,192,(192,2),192,384,384,384,384,384,384,384,384,576,576,576,576,576,576,576,576,(576,2),576,960,960,960,960, 960,960 ]

        k=0
        for i in range(0,len(self.cfg),2):
            in_planes = self.cfg_res[k]
            cfg0 = self.cfg[i ] if isinstance(self.cfg[i], int) else self.cfg[i][0]
            cfg1 = self.cfg[i + 1] if isinstance(self.cfg[i+1], int) else self.cfg[i + 1][0]
            out_planes = self.cfg_res[k+1]
            stride = 1 if isinstance(self.cfg[i], int) else self.cfg[i][1]

            layers.append(Block(in_planes, out_planes, cfg0,cfg1, stride))
            k+=1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)

