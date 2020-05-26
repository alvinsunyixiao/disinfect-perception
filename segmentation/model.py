import math
import torch

from torch import nn
from torchvision.models import resnet
from torchvision.models.utils import load_state_dict_from_url

from utils.params import ParamDict as o

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
        self.bn = nn.BatchNorm2d(out_channel) if has_bn else nn.Identity()
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

        return (x4, x3, x2, x1)

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

        return tuple(fmaps)

class SegHead(nn.Module):

    def __init__(self, in_channel, mid_channel, num_layers, num_classes, forground=0.01):
        super(SegHead, self).__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_channel if i == 0 else mid_channel
            layers.append(SameConvBNReLU(in_ch, mid_channel, 3, 1))
        self.convs = nn.Sequential(*layers)
        self.reduce = nn.Conv2d(mid_channel, num_classes, 1, 1)
        nn.init.normal_(self.reduce.weight, std=1e-2)
        nn.init.constant_(self.reduce.bias,
                          -math.log((1-forground) / forground))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.convs(x)
        x = self.reduce(x)
        x = self.upsample(x)
        x = self.output_sigmoid(x)
        return x

class FPNx(nn.Module):

    def __init__(self, backbone, fp_channels, seg_channels, seg_layers,
                 num_classes, backbone_channels, pretrianed_backbone=True):
        super(FPNx, self).__init__()
        self.backbone = backbone(pretrained=pretrianed_backbone)
        self.fpn = FeaturePyramid(backbone_channels, fp_channels)
        self.seghead = SegHead(fp_channels, seg_channels, seg_layers, num_classes)

    def forward(self, x):
        backbone_layers = self.backbone(x)
        feature_layers = self.fpn(*backbone_layers)
        output = []
        for fmap in feature_layers:
            output.append(self.seghead(fmap))

        return tuple(output)

class FPNResNet18(FPNx):

    DEFAULT_PARAMS=o(
        fp_channels=256,
        seg_channels=64,
        seg_layers=1,
        num_classes=21,
        pretrianed_backbone=True,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params
        super(FPNResNet18, self).__init__(
            backbone=ResNet18,
            fp_channels=self.p.fp_channels,
            seg_channels=self.p.seg_channels,
            seg_layers=self.p.seg_layers,
            num_classes=self.p.num_classes,
            backbone_channels=(512, 256, 128, 64),
            pretrianed_backbone=self.p.pretrianed_backbone,
        )

