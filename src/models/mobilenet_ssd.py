import mlconfig
from torch import nn
import torch

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)

class DepthwiseConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride=1):
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(DepthwiseConv, self).__init__(*layers)

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, shallow=False):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(*self.get_layers())
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(nearby_int(width_mult * 1024), num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def get_layers():
        settings = [
            (32, 2),
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        layers = []
        in_channels = 3
        for i, (out_channels, stride) in enumerate(settings):
            if i == 0:
                layers += [ConvBNReLU(in_channels, out_channels, stride=stride)]
            else:
                layers += [DepthwiseConv(in_channels, out_channels, stride=stride)]
            in_channels = out_channels
        return layers

class ExtraBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        hidden_dim = out_channels // 2
        layers = [
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ExtraBlock, self).__init__(*layers)

class HeadBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, additional_conv=False):
        hidden_dim = in_channels if additional_conv else in_channels
        layers = []
        if additional_conv:
            layers += [DepthwiseConv(in_channels, hidden_dim)]
        layers += [nn.Conv2d(hidden_dim, out_channels, 3, padding=1)]
        super(HeadBlock, self).__init__(*layers)
        
class L2Norm(nn.Module):

    def __init__(self, in_channels, gamma=1.0, eps=1e-10):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.gamma = gamma
        self.eps = eps
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = nn.Parameter(torch.Tensor(1, self.in_channels, 1, 1)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weights, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        out = self.weights * torch.div(x, norm + self.eps)
        return out
        
@mlconfig.register
class MobileNet_SSD(nn.Module):

    def __init__(self, num_classes=21, boxes = [4,6,6,6,4,4]):
        super(MobileNet_SSD, self).__init__()
        self.num_classes = num_classes
        self.boxes = boxes
        
        self.features = nn.ModuleList(MobileNet.get_layers() + [ExtraBlock(1024, 512),
                                                    ExtraBlock(512, 256),
                                                    ExtraBlock(256, 128)])

        self.classifier = nn.ModuleList([
            HeadBlock(256, boxes[0] * num_classes, True),
            HeadBlock(512, boxes[1] * num_classes, True),
            HeadBlock(1024, boxes[2] * num_classes, False),
            HeadBlock(512, boxes[3] * num_classes, False),
            HeadBlock(256, boxes[4] * num_classes, False),
            HeadBlock(128, boxes[5] * num_classes, False),
        ])

        self.regression = nn.ModuleList([
            HeadBlock(256, boxes[0] * 4, True),
            HeadBlock(512, boxes[1] * 4, True),
            HeadBlock(1024, boxes[2] * 4, False),
            HeadBlock(512, boxes[3] * 4, False),
            HeadBlock(256, boxes[4] * 4, False),
            HeadBlock(128, boxes[5] * 4, False),
        ])
        
        self._initialize_weights()
        
    def forward(self, x):
        sources = []
        for i, v in enumerate(self.features):
            x = v(x)
            if i in [5, 11, 13, 14, 15, 16]:
                sources.append(x)

        l2norm = L2Norm(256)
        sources[0] = l2norm(sources[0])

        # ssd
        conf = []
        loc = []
        for s, op in zip(sources, self.classifier):
            conf.append(op(s).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes))

        for s, op in zip(sources, self.regression):
            loc.append(op(s).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4))

        conf = torch.cat(conf, dim=1)
        loc = torch.cat(loc, dim=1)
        return loc, conf
        
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
