import math
import torch

from torch import nn
from torchvision.models import resnet

from utils.params import ParamDict as o

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

    def __init__(self, arch='resnet50', **kwargs):
        super(ResNet, self).__init__(**kwargs)
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

    def __init__(self, **kwargs):
        super(ResNet50, self).__init__(
            arch='resnet50',
            block=resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            **kwargs,
        )

class ResNet18(ResNet):

    def __init__(self, **kwargs):
        super(ResNet18, self).__init__(
            arch='resnet18',
            block=resnet.BasicBlock,
            layers=[2, 2, 2, 2],
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

    def forward(self, inputs):
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
        self.backbone = backbone()
        self.fpn = FeaturePyramid(backbone_channels, fp_channels)
        self.seghead = SegHead(fp_channels, seg_channels, seg_layers, num_classes)

    def forward(self, x):
        backbone_layers = self.backbone(x)
        feature_layers = self.fpn(backbone_layers)
        output = []
        for fmap in feature_layers:
            output.append(self.seghead(fmap))

        return tuple(output)

class FPNResNet18(FPNx):

    DEFAULT_PARAMS=o(
        fp_channels=256,
        seg_channels=256,
        seg_layers=1,
        num_classes=21,
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
        )

# late night code
# TODO: separate these out to an isolated file

from mit_semseg.models.models import mobilenet, MobileNetV2Dilated
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=256, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.interpolate(
                x, size=256, mode='bilinear', align_corners=False)
            x = nn.functional.log_softmax(x, dim=1)

        return x

class NoFPNMobileNetV2Dilated(nn.Module):

    DEFAULT_PARAMS=o(
        dilate_scale=8,
    )

    def __init__(self, params = DEFAULT_PARAMS):
        super(NoFPNMobileNetV2Dilated, self).__init__()
        self.p = params
        orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=True)
        self.backbone = MobileNetV2Dilated(orig_mobilenet, dilate_scale=self.p.dilate_scale)
        self.net_decoder = C1(
                num_class=150,
                fc_dim=320,
                use_softmax=False)
    
    def forward(self, x):
        return self.net_decoder(self.backbone(x, return_feature_maps = True))