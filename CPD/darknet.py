import torch
import torch.nn as nn
import numpy as np

class D19(nn.Module):
    def __init__(self):
        super(D19, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        conv1.add_module('bn1_1', nn.BatchNorm2d(32))
        conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('maxpool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        conv2.add_module('bn2_1', nn.BatchNorm2d(64))
        conv2.add_module('relu2_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('maxpool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        conv3.add_module('bn3_1', nn.BatchNorm2d(128))
        conv3.add_module('relu3_1', nn.LeakyReLU(0.1, inplace=True))
        conv3.add_module('conv3_2', nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        conv3.add_module('bn3_2', nn.BatchNorm2d(64))
        conv3.add_module('relu3_2', nn.LeakyReLU(0.1, inplace=True))
        conv3.add_module('conv3_3', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        conv3.add_module('bn3_3', nn.BatchNorm2d(128))
        conv3.add_module('relu3_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = conv3

        conv4_1 = nn.Sequential()
        conv4_1.add_module('maxpool4_1', nn.MaxPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_1.add_module('bn4_1_1', nn.BatchNorm2d(256))
        conv4_1.add_module('relu4_1_1', nn.LeakyReLU(0.1, inplace=True))
        conv4_1.add_module('conv4_1_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        conv4_1.add_module('bn4_1_2', nn.BatchNorm2d(128))
        conv4_1.add_module('relu4_1_2', nn.LeakyReLU(0.1, inplace=True))
        conv4_1.add_module('conv4_1_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_1.add_module('bn4_1_3', nn.BatchNorm2d(256))
        conv4_1.add_module('relu4_1_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_1', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_1', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_1.add_module('bn5_1_2', nn.BatchNorm2d(256))
        conv5_1.add_module('relu5_1_2', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_3', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_3', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_1.add_module('bn5_1_4', nn.BatchNorm2d(256))
        conv5_1.add_module('relu5_1_4', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_5', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_5', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_1 = conv5_1

        conv4_2 = nn.Sequential()
        conv4_2.add_module('maxpool4_2', nn.MaxPool2d(2, stride=2))
        conv4_2.add_module('conv4_2_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_2.add_module('bn4_2_1', nn.BatchNorm2d(256))
        conv4_2.add_module('relu4_2_1', nn.LeakyReLU(0.1, inplace=True))
        conv4_2.add_module('conv4_2_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        conv4_2.add_module('bn4_2_2', nn.BatchNorm2d(128))
        conv4_2.add_module('relu4_2_2', nn.LeakyReLU(0.1, inplace=True))
        conv4_2.add_module('conv4_2_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_2.add_module('bn4_2_3', nn.BatchNorm2d(256))
        conv4_2.add_module('relu4_2_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_2 = conv4_2

        conv5_2 = nn.Sequential()
        conv5_2.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        conv5_2.add_module('conv5_2_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_2.add_module('bn5_2_1', nn.BatchNorm2d(512))
        conv5_2.add_module('relu5_2_1', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_2.add_module('bn5_2_2', nn.BatchNorm2d(256))
        conv5_2.add_module('relu5_2_2', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_2.add_module('bn5_2_3', nn.BatchNorm2d(512))
        conv5_2.add_module('relu5_2_3', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_2.add_module('bn5_2_4', nn.BatchNorm2d(256))
        conv5_2.add_module('relu5_2_4', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_2.add_module('bn5_2_5', nn.BatchNorm2d(512))
        conv5_2.add_module('relu5_2_5', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_2 = conv5_2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)

        # Attention branch
        x4_1 = self.conv4_1(x3)
        x5_1 = self.conv5_1(x4_1)

        # Detection branch
        x4_2 = self.conv4_2(x3)
        x5_2 = self.conv5_2(x4_2)

        return x3, x4_1, x5_1, x4_1, x5_1


class D19_new(nn.Module):
    def __init__(self):
        super(D19_new, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        self.conv1.add_module('bn1_1', nn.BatchNorm2d(32))
        self.conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('maxpool1', nn.MaxPool2d(2, stride=2))
        self.conv2.add_module('conv2_1', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        self.conv2.add_module('bn2_1', nn.BatchNorm2d(64))
        self.conv2.add_module('relu2_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('maxpool2', nn.MaxPool2d(2, stride=2))
        self.conv3.add_module('conv3_1', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        self.conv3.add_module('bn3_1', nn.BatchNorm2d(128))
        self.conv3.add_module('relu3_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv3.add_module('conv3_2', nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        self.conv3.add_module('bn3_2', nn.BatchNorm2d(64))
        self.conv3.add_module('relu3_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv3.add_module('conv3_3', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        self.conv3.add_module('bn3_3', nn.BatchNorm2d(128))
        self.conv3.add_module('relu3_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv4_1 = nn.Sequential()
        self.conv4_1.add_module('maxpool4_1', nn.MaxPool2d(2, stride=2))
        self.conv4_1.add_module('conv4_1_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4_1.add_module('bn4_1_1', nn.BatchNorm2d(256))
        self.conv4_1.add_module('relu4_1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_1.add_module('conv4_1_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        self.conv4_1.add_module('bn4_1_2', nn.BatchNorm2d(128))
        self.conv4_1.add_module('relu4_1_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_1.add_module('conv4_1_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4_1.add_module('bn4_1_3', nn.BatchNorm2d(256))
        self.conv4_1.add_module('relu4_1_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv5_1 = nn.Sequential()
        self.conv5_1.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        self.conv5_1.add_module('conv5_1_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5_1.add_module('bn5_1_1', nn.BatchNorm2d(512))
        self.conv5_1.add_module('relu5_1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_1.add_module('conv5_1_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5_1.add_module('bn5_1_2', nn.BatchNorm2d(256))
        self.conv5_1.add_module('relu5_1_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_1.add_module('conv5_1_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5_1.add_module('bn5_1_3', nn.BatchNorm2d(512))
        self.conv5_1.add_module('relu5_1_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv4_2 = nn.Sequential()
        self.conv4_2.add_module('maxpool4_2', nn.MaxPool2d(2, stride=2))
        self.conv4_2.add_module('conv4_2_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4_2.add_module('bn4_2_1', nn.BatchNorm2d(256))
        self.conv4_2.add_module('relu4_2_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_2.add_module('conv4_2_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        self.conv4_2.add_module('bn4_2_2', nn.BatchNorm2d(128))
        self.conv4_2.add_module('relu4_2_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_2.add_module('conv4_2_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4_2.add_module('bn4_2_3', nn.BatchNorm2d(256))
        self.conv4_2.add_module('relu4_2_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv5_2 = nn.Sequential()
        self.conv5_2.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        self.conv5_2.add_module('conv5_2_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5_2.add_module('bn5_2_1', nn.BatchNorm2d(512))
        self.conv5_2.add_module('relu5_2_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_2.add_module('conv5_2_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5_2.add_module('bn5_2_2', nn.BatchNorm2d(256))
        self.conv5_2.add_module('relu5_2_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_2.add_module('conv5_2_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5_2.add_module('bn5_2_3', nn.BatchNorm2d(512))
        self.conv5_2.add_module('relu5_2_3', nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)

        # Attention branch
        x4_1 = self.conv4_1(x3)
        x5_1 = self.conv5_1(x4_1)

        # Detection branch
        x4_2 = self.conv4_2(x3)
        x5_2 = self.conv5_2(x4_2)

        return x3, x4_1, x5_1, x4_1, x5_1


class D19_A(nn.Module):
    def __init__(self):
        super(D19_A, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        self.conv1.add_module('bn1_1', nn.BatchNorm2d(32))
        self.conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('maxpool1', nn.MaxPool2d(2, stride=2))
        self.conv2.add_module('conv2_1', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        self.conv2.add_module('bn2_1', nn.BatchNorm2d(64))
        self.conv2.add_module('relu2_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('maxpool2', nn.MaxPool2d(2, stride=2))
        self.conv3.add_module('conv3_1', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        self.conv3.add_module('bn3_1', nn.BatchNorm2d(128))
        self.conv3.add_module('relu3_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv3.add_module('conv3_2', nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        self.conv3.add_module('bn3_2', nn.BatchNorm2d(64))
        self.conv3.add_module('relu3_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv3.add_module('conv3_3', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        self.conv3.add_module('bn3_3', nn.BatchNorm2d(128))
        self.conv3.add_module('relu3_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('maxpool4_1', nn.MaxPool2d(2, stride=2))
        self.conv4.add_module('conv4_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4.add_module('bn4_1', nn.BatchNorm2d(256))
        self.conv4.add_module('relu4_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv4.add_module('conv4_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        self.conv4.add_module('bn4_2', nn.BatchNorm2d(128))
        self.conv4.add_module('relu4_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv4.add_module('conv4_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4.add_module('bn4_3', nn.BatchNorm2d(256))
        self.conv4.add_module('relu4_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        self.conv5.add_module('conv5_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5.add_module('bn5_1', nn.BatchNorm2d(512))
        self.conv5.add_module('relu5_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv5.add_module('conv5_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5.add_module('bn5_2', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv5.add_module('conv5_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5.add_module('bn5_3', nn.BatchNorm2d(512))
        self.conv5.add_module('relu5_3', nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x3, x4, x5


class D19_A_pruned(nn.Module):
    def __init__(self):
        super(D19_A_pruned, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_1', nn.Conv2d(3, 26, 3, 1, 1, bias=False))
        self.conv1.add_module('bn1_1', nn.BatchNorm2d(26))
        self.conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('avgpool1', nn.AvgPool2d(2, stride=2))
        self.conv2.add_module('conv2_1', nn.Conv2d(26, 64, 3, 1, 1, bias=False))
        self.conv2.add_module('bn2_1', nn.BatchNorm2d(64))
        self.conv2.add_module('relu2_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('avgpool2', nn.AvgPool2d(2, stride=2))
        self.conv3.add_module('conv3_1', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        self.conv3.add_module('bn3_1', nn.BatchNorm2d(128))
        self.conv3.add_module('relu3_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv3.add_module('conv3_2', nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        self.conv3.add_module('bn3_2', nn.BatchNorm2d(64))
        self.conv3.add_module('relu3_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv3.add_module('conv3_3', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        self.conv3.add_module('bn3_3', nn.BatchNorm2d(128))
        self.conv3.add_module('relu3_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('avgpool4_1', nn.AvgPool2d(2, stride=2))
        self.conv4.add_module('conv4_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4.add_module('bn4_1', nn.BatchNorm2d(256))
        self.conv4.add_module('relu4_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv4.add_module('conv4_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        self.conv4.add_module('bn4_2', nn.BatchNorm2d(128))
        self.conv4.add_module('relu4_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv4.add_module('conv4_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        self.conv4.add_module('bn4_3', nn.BatchNorm2d(256))
        self.conv4.add_module('relu4_3', nn.LeakyReLU(0.1, inplace=True))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('avgpool5_1', nn.AvgPool2d(2, stride=2))
        self.conv5.add_module('conv5_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5.add_module('bn5_1', nn.BatchNorm2d(512))
        self.conv5.add_module('relu5_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv5.add_module('conv5_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5.add_module('bn5_2', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv5.add_module('conv5_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5.add_module('bn5_3', nn.BatchNorm2d(512))
        self.conv5.add_module('relu5_3', nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x3, x4, x5


if __name__ == '__main__':
    model = D19_new()
