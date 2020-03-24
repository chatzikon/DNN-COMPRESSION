'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes1,out_planes2, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes1, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes1)
        self.conv2 = nn.Conv2d(out_planes1, out_planes2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    #cfg = [32,32,64, (64,2), 128, 128, 128, (128,2), 256, 256,256, (256,2),512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, (512,2), 1024, 1024, 1024]
    cfg = [32, 32, 64, 64, 128, 128, 128, 128, 256, 256, 256, (256,2), 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, (512,2), 1024, 1024, 1024]

    def __init__(self, num_classes=10,cfg=None):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(cfg)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(1024, num_classes)

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
        print('selfcfg')
        print(self.cfg)


        for i in range(1,len(self.cfg),2):
            in_planes = self.cfg[i-1] if isinstance(self.cfg[i-1], int) else self.cfg[i-1][0]
            out_planes1 = self.cfg[i ] if isinstance(self.cfg[i], int) else self.cfg[i][0]
            out_planes2 = self.cfg[i + 1] if isinstance(self.cfg[i+1], int) else self.cfg[i + 1][0]
            stride = 1 if isinstance(self.cfg[i], int) else self.cfg[i][1]
            # print('planes')
            # print(in_planes)
            # print(out_planes1)
            # print(out_planes2)
            layers.append(Block(in_planes,out_planes1,out_planes2, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def test():
#     net = MobileNet()
#     x = torch.randn(1,3,32,32)
#     y = net(x)
#     print(y.size())

# test()
