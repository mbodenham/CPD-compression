import torch
import torch.nn as nn
import numpy as np

class D19_D(nn.Module):
    def __init__(self):
        super(D19_D, self).__init__()
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
        self.conv5_2.add_module('conv5_2_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5_2.add_module('bn5_2_4', nn.BatchNorm2d(256))
        self.conv5_2.add_module('relu5_2_4', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_2.add_module('conv5_2_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5_2.add_module('bn5_2_5', nn.BatchNorm2d(512))
        self.conv5_2.add_module('relu5_2_5', nn.LeakyReLU(0.1, inplace=True))

        weights = torch.load('./CPD/darknet19_weights.pth')
        self._initialize_weights(weights)

    def forward(self, x3):
        x4_2 = self.conv4_2(x3)
        x5_2 = self.conv5_2(x4_2)

        return x4_2, x5_2

    def _initialize_weights(self, weights):
        keys = list(weights.keys())
        i = 30
        self.conv4_2.conv4_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.conv4_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.conv4_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_2.bn4_2_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5_2.conv5_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.conv5_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.conv5_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.conv5_2_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_4.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_4.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_4.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_4.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.conv5_2_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_5.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_5.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_5.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_2.bn5_2_5.num_batches_tracked.data.copy_(weights[keys[i]])



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
        self.conv5.add_module('conv5_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5.add_module('bn5_4', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_4', nn.LeakyReLU(0.1, inplace=True))
        self.conv5.add_module('conv5_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5.add_module('bn5_5', nn.BatchNorm2d(512))
        self.conv5.add_module('relu5_5', nn.LeakyReLU(0.1, inplace=True))

        weights = torch.load('./CPD/darknet19_weights.pth')
        self._initialize_weights(weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x3, x4, x5

    def _initialize_weights(self, weights):
        keys = list(weights.keys())
        i = 0
        self.conv1.conv1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv2.conv2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv3.conv3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv4.conv4_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv4.conv4_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.conv4_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5.conv5_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.conv5_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5.conv5_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.num_batches_tracked.data.copy_(weights[keys[i]])

        i+=1
        self.conv5.conv5_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.conv5_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.num_batches_tracked.data.copy_(weights[keys[i]])

class D19_A_avg(nn.Module):
    def __init__(self):
        super(D19_A_avg, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        self.conv1.add_module('bn1_1', nn.BatchNorm2d(32))
        self.conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('avgpool1', nn.AvgPool2d(2, stride=2))
        self.conv2.add_module('conv2_1', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
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
        self.conv5.add_module('conv5_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5.add_module('bn5_4', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_4', nn.LeakyReLU(0.1, inplace=True))
        self.conv5.add_module('conv5_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5.add_module('bn5_5', nn.BatchNorm2d(512))
        self.conv5.add_module('relu5_5', nn.LeakyReLU(0.1, inplace=True))

        weights = torch.load('./CPD/darknet19_weights.pth')
        self._initialize_weights(weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x3, x4, x5

    def _initialize_weights(self, weights):
        keys = list(weights.keys())
        i = 0
        self.conv1.conv1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv2.conv2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv3.conv3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv4.conv4_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv4.conv4_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.conv4_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5.conv5_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.conv5_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5.conv5_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.num_batches_tracked.data.copy_(weights[keys[i]])

        i+=1
        self.conv5.conv5_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.conv5_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.num_batches_tracked.data.copy_(weights[keys[i]])

class D19_A_P(nn.Module):
    def __init__(self, out_channel=32):
        super(D19_A_P, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_1', nn.Conv2d(3, out_channel, 3, 1, 1, bias=False))
        self.conv1.add_module('bn1_1', nn.BatchNorm2d(out_channel))
        self.conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('avgpool1', nn.AvgPool2d(2, stride=2))
        self.conv2.add_module('conv2_1', nn.Conv2d(out_channel, 64, 3, 1, 1, bias=False))
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
        self.conv5.add_module('conv5_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        self.conv5.add_module('bn5_4', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_4', nn.LeakyReLU(0.1, inplace=True))
        self.conv5.add_module('conv5_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        self.conv5.add_module('bn5_5', nn.BatchNorm2d(512))
        self.conv5.add_module('relu5_5', nn.LeakyReLU(0.1, inplace=True))

        weights = torch.load('./CPD/darknet19_weights.pth')
        self._initialize_weights(weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x3, x4, x5

    def _initialize_weights(self, weights):
        keys = list(weights.keys())
        i = 0
        #self.conv1.conv1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        #self.conv1.bn1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        #self.conv1.bn1_1.bias.data.copy_(weights[keys[i]])
        i+=1
        #self.conv1.bn1_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        #self.conv1.bn1_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        #self.conv1.bn1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        #self.conv2.conv2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv3.conv3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv4.conv4_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv4.conv4_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.conv4_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4.bn4_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5.conv5_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.conv5_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5.conv5_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_3.num_batches_tracked.data.copy_(weights[keys[i]])

        i+=1
        self.conv5.conv5_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_4.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.conv5_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5.bn5_5.num_batches_tracked.data.copy_(weights[keys[i]])


if __name__ == '__main__':
    model = D19_new()
