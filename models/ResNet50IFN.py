from .backbones.resnetifn import Bottleneck, init_pretrained_weights, model_urls
from torch import nn
from .BasicModule import BasicModule

def wavg(feat_map):
    batch = feat_map.shape[0]
    channel = feat_map.shape[1]

    return ((feat_map**2).view(batch, channel, -1) / (1e-12 + feat_map.view(batch, channel, -1).sum(-1, keepdim=True))).sum(-1)


class ResNet50IFN(BasicModule):
    """Residual network.

    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes,
                 last_stride,
                 pooling,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 fc_dims=None,
                 dropout_p=None,
                 **kwargs):
        self.inplanes = 64
        super(ResNet50IFN, self).__init__()
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],IN=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,IN=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,IN=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        if pooling == 'AVG':
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'MAX':
            self.global_avgpool = nn.AdaptiveMaxPool2d(1)
        elif pooling == 'WAVG':
            self.global_avgpool = wavg
        else:
            raise Exception('The POOL value should be AVG or MAX or WAVG')

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        self._init_params()
        init_pretrained_weights(self, model_urls['resnet50'])

    def _make_layer(self, block, planes, blocks, stride=1,IN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes,planes,IN=IN))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer
        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        g_v = v.view(v.size(0), -1)

        v = self.bottleneck(g_v)

        if not self.training:
            return v

        y = self.classifier(v)
        return (y,), (g_v,)
