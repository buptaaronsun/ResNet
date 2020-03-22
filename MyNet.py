from __future__ import print_function
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(planes),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(planes),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Conv2d(planes, 4*planes, kernel_size=1,stride=1),
                                    nn.BatchNorm2d(4*planes),
                                    )
        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.downsample is not None:
            res = self.downsample(res)

        x += res

        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.layer0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size=3, stride=2)
                                    )
        self.layer1 = self._make_layer(64,3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()


    def _make_layer(self, planes, times, stride=1):
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        else:
            downsample = None
        layers = []
        layers.append(ResBlock(in_planes=self.inplanes, planes=planes, stride=stride,downsample=downsample))
        self.inplanes=planes*4
        for i in range(1,times):
            layers.append(ResBlock(in_planes=self.inplanes,planes=planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x =x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def feature_vector(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x




