from torch import nn
from .BasicModule import BasicModule
from torchvision import models
from .backbones.resnet import resnet50

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def wavg(feat_map):
    batch = feat_map.shape[0]
    channel = feat_map.shape[1]

    return ((feat_map**2).view(batch, channel, -1) / (1e-12 + feat_map.view(batch, channel, -1).sum(-1, keepdim=True))).sum(-1)

class Generalizer_G(BasicModule):
    in_planes = 2048
    def __init__(self, num_classes, last_stride, pooling):
        super().__init__()
        self.model_name = 'ResNet50'
        self.base = resnet50(pretrained=True, last_stride=last_stride)
        if pooling == 'AVG':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'MAX':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pooling == 'WAVG':
            self.pool = wavg
        else:
            raise Exception('The POOL value should be AVG or MAX or WAVG')
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        # self.bottleneck_gan = nn.BatchNorm1d(self.in_planes)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        # self.bottleneck_gan.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, img):
        # x is the feature map. h is the encoded feature map
        if self.training:
            feature_map = self.base(img)  # [batch, 2048, 12, 4]
            feature_vec = self.pool(feature_map).view(img.shape[0], -1)
            feature_vec_o = self.bottleneck(feature_vec)
            scores = self.classifier(feature_vec_o)
            # gan_vec = self.bottleneck_gan(feature_vec)
            gan_vec = feature_vec

            return feature_vec, scores, gan_vec
        else:
            feature_map = self.base(img)
            feature_vec = self.pool(feature_map).view(img.shape[0], -1)

            return feature_vec


class Generalizer_D(BasicModule):
    def __init__(self, num_domains):
        super().__init__()

        self.valid = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_domains)  # 12-2 * 4-2 = 10 * 2
        )

    def forward(self, x):
        validity = self.valid(x)

        return validity
