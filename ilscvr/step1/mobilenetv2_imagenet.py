from torch import nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp,  stride,i,cfg1,cfg2,cfg3):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]




        self.use_res_connect = self.stride == 1 and inp == cfg3

        layers = []
        if i> 0:
            # pw
            layers.append(ConvBNReLU(inp, cfg1, kernel_size=1))
            layers.extend([
                # dw
                ConvBNReLU(cfg1, cfg2, stride=stride, groups=cfg1),
                # pw-linear
                nn.Conv2d(cfg2, cfg3, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cfg3),
            ])
        else:
            layers.extend([
                # dw
                ConvBNReLU(inp, cfg2, stride=stride, groups=inp),
                # pw-linear
                nn.Conv2d(cfg2, cfg3, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cfg3),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    inverted_residual_setting = [0, 32, 96, (96, 2), 144, 144, 144, (144, 2), 192, 192, 192, 192,
                                 192, (192, 2), 384, 384, 384, 384, 384, 384, 384, 384, 576, 576, 576, 576, 576,
                                 (576, 2), 960, 960, 960, 960, 960, 960]
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        self.last_channel = 1280

        if inverted_residual_setting is None:
            self.inverted_residual_setting = [0,32,  96, (96, 2),  144, 144,  144, (144, 2),  192, 192,  192, 192,
                                         192,(192, 2), 384, 384, 384, 384,  384, 384,  384, 384,  576, 576,  576, 576,  576,
                                         (576, 2), 960, 960, 960, 960,  960, 960]
        else:
            for i in range(len(inverted_residual_setting)):
                if isinstance(self.inverted_residual_setting[i], int):
                    self.inverted_residual_setting[i]=inverted_residual_setting[i]
                else:
                    temp=list(self.inverted_residual_setting[i])
                    temp[0]=inverted_residual_setting[i]
                    self.inverted_residual_setting[i] = tuple(temp)



        inverted_residual_setting1=[16,24,24, 32,32,32,64,64,64,64,96,96,96,160,160,160,320]











        features = [ConvBNReLU(3, input_channel, stride=2)]

        k=0
        for i in range(0, len(self.inverted_residual_setting),2):
                stride = 1 if isinstance(self.inverted_residual_setting[i+1], int) else self.inverted_residual_setting[i+1][1]
                cfg1 = self.inverted_residual_setting[i + 1] if isinstance(self.inverted_residual_setting[i + 1], int) else self.inverted_residual_setting[i + 1][0]
                features.append(block(input_channel, stride,i,self.inverted_residual_setting[i],cfg1,inverted_residual_setting1[k]))
                input_channel = inverted_residual_setting1[k]
                k+=1
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model