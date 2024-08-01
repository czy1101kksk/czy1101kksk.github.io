import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1, groups=1, dilation=1):
        super(Residual, self):__init__()
        self.Batch1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, padding=1, bias=False)
        self.Batch2 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=strides, padding=1, bias=False)

    def forward(self, x):
        identity = x
        
        out = self.Batch1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.Batch2(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity
        out = self.relu(out)
        
        return out



class Bottleneck(nn.Module):
    
    def __init__(self, input_channels, num_channels, strides=1, groups=1, dilation=1):
        super(Bottleneck, self):__init__()
        self.expansion = 4
        self.Batch1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, padding=1, bias=False)
        self.Batch2 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=strides, padding=1, bias=False)
        self.Batch3 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=strides, padding=1, bias=False)

    def forward(self, x):
        identity = x

        out = self.Batch1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.Batch2(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.Batch3(x)
        out = self.relu(out)
        out = self.conv3(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, ,num_classes=1000, groups=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = block(64, 128,stride=2,)
        self.layer2 = block(128, 256,stride=2,)
        self.layer3 = block(256, 512,stride=2,)               
        self.layer4 = block(512, 512 * 4,stride=2,)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
     

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

   def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

