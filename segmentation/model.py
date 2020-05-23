import math
import torch

from torch import nn
from torchvision.models import resnet
from torchvision.models.utils import load_state_dict_from_url

class ImageNormalize(nn.Module):

    def __init__(self, mean, std):
        super(ImageNormalize, self).__init__()
        self.mean_1311 = nn.Parameter(torch.FloatTensor(mean)[None, :, None, None],
                                      requires_grad=False)
        self.std_1311 = nn.Parameter(torch.FloatTensor(std)[None, :, None, None],
                                     requires_grad=False)

    def forward(self, x):
        return (x - self.mean_1311) / self.std_1311

class SameConvBNReLU(nn.ModuleDict):
    """ Same padded conv2d with optional bn and relu """

    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 has_bn=True, has_relu=True):
        super(SameConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=(not has_bn),
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.bn = nn.BatchNorm2d(in_channel) if has_bn else nn.Identity()
        self.relu = nn.ReLU() if has_relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResNet(resnet.ResNet):
    """ Modified ResNet backbone implementation for FPN structures
    @see https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
    """

    def __init__(self, arch='resnet50', pretrained=True, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(resnet.model_urls[arch],
                                                  progress=True)
            self.load_state_dict(state_dict)
        self.normalize = ImageNormalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    def _forward_impl(self, x):
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1]

class ResNet50(ResNet):

    def __init__(self, pretrained=True, **kwargs):
        super(ResNet50, self).__init__(
            arch='resnet50',
            block=resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            pretrained=pretrained,
            **kwargs,
        )

class ResNet18(ResNet):

    def __init__(self, pretrained=True, **kwargs):
        super(ResNet18, self).__init__(
            arch='resnet18',
            block=resnet.BasicBlock,
            layers=[2, 2, 2, 2],
            pretrained=pretrained,
            **kwargs,
        )

class FeaturePyramid(nn.Module):

    def __init__(self, in_channels, out_channel):
        super(FeaturePyramid, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.reduce = []
        for i, in_channel in enumerate(in_channels):
            conv = nn.Conv2d(in_channel, out_channel, 1)
            nn.init.normal_(conv.weight, std=1e-2)
            nn.init.zeros_(conv.bias)
            self.reduce.append(conv)
            self.add_module('reduce{}'.format(i), conv)

    def forward(self, *inputs):
        assert len(inputs) == len(self.reduce), "Feature pyramid size mismatch"
        fmaps = []
        for i, redu in enumerate(self.reduce):
            reduced_fmap = redu(inputs[i])
            if i == 0:
                merged_fmap = reduced_fmap
            else:
                merged_fmap = reduced_fmap + self.upsample(prev_fmap)
            fmaps.append(merged_fmap)
            prev_fmap = merged_fmap

        return fmaps

class SegHead(nn.Module):

    def __init__(self, in_channel, num_layers, num_classes, forground=0.01):
        super(SegHead, self).__init__()
        layers = [SameConvBNReLU(in_channel, in_channel, 3, 1) for _ in range(num_layers)]
        self.head = nn.Sequential(*layers)
        self.upsample = nn.ConvTranspose2d(in_channel, num_classes, 2, 2)
        nn.init.normal_(self.upsample.weight, std=1e-2)
        # see focal loss for this initialization
        nn.init.constant_(self.upsample.bias,
                          -math.log((1-forground) / forground))
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.head(x)
        x = self.upsample(x)
        x = self.output_sigmoid(x)
        return x

class FPNResNetx(nn.Module):

    def __init__(self, resnet, fp_channels, num_classes,
                 backbone_channels, pretrianed_backbone=True):
        super(FPNResNetx, self).__init__()
        self.resnet = resnet(pretrained=pretrianed_backbone)
        self.fpn = FeaturePyramid(backbone_channels, fp_channels)
        self.seghead = SegHead(fp_channels, 1, num_classes)

    def forward(self, x):
        backbone_layers = self.resnet(x)
        feature_layers = self.fpn(*backbone_layers)
        output = []
        for fmap in feature_layers:
            output.append(self.seghead(fmap))

        return output

class FPNResNet18(FPNResNetx):

    def __init__(self, fp_channels, num_classes, pretrianed_backbone=True):
        super(FPNResNet18, self).__init__(
            resnet=ResNet18,
            fp_channels=fp_channels,
            num_classes=num_classes,
            backbone_channels=(512, 256, 128, 64),
            pretrianed_backbone=pretrianed_backbone,
        )

