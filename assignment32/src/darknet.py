import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels,dilated=1):
        super(Residual, self).__init__()
        mid_channels = in_channels // 2
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels,out_channels=in_channels,
                      kernel_size=3, dilation=dilated, padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,
                      kernel_size=1, dilation=dilated,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.conv33(self.conv11(x))
        return out + x

class DarkNet(nn.Module):
    def __init__(self, S, B, cls):
        super(DarkNet, self).__init__()
        output_size = S *S *(cls+B*5)
        self.S = S
        self.B = B
        self.cls = cls
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,
                      kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.layer1 = self._make_layers(in_channels=32,out_channels=64, block_num=1,stride=2)
        self.layer2 = self._make_layers(in_channels=64,out_channels=128, block_num=2,stride=2)
        self.layer3 = self._make_layers(in_channels=128,out_channels=256, block_num=8,stride=2)
        self.layer4 = self._make_layers(in_channels=256,out_channels=512, block_num=8,stride=2)
        self.layer5 = self._make_layers(in_channels=512,out_channels=1024, block_num=4,stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=1024,out_features=output_size)
        self.softmax = nn.Softmax(dim=1)
        self.conv_end = nn.Conv2d(
            1024, B * 5 + 20, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_end = nn.BatchNorm2d(B * 5 + 20)

    def _make_layers(self, in_channels,out_channels, block_num, dilated=1, stride=1):
        _layers = []
        
        _layers.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                      kernel_size=3,stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        ))
        for _ in range(block_num):
            _layers.append(Residual(in_channels=out_channels,dilated=dilated))
        return nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.conv_end(x)
        x = self.bn_end(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        # x = self.avg_pool(x)
        # x = x.view(x.size(0),-1)
        # x = self.linear(x)
        # x = self.softmax(x)
        # out = x.reshape((x.shape[0], self.S, self.S, (self.B*5 + self.cls)))
        return x

def update_state_dict(pretrained_model, model):
    new_state_dict = pretrained_model.state_dict()
    dd = model.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith("fc"):
            dd[k] = new_state_dict[k]
    model.load_state_dict(dd)
    return model

def darknet53(S, B, cls, pretrained=None):
    """Constructs a DarkNet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet(S, B, cls)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
    return model
