import torch
import torch.nn as nn

from .darknet import D19_D, D19_A, D19_A_avg, D19_A_P
from .evaluate import Eval
from .dataset import EvalImageGroundTruthFolder, ImageGroundTruthFolder
from .modules import aggregation, HA, RFB, aggregation_pruned, RFB_pruned
from .vgg import VGG_D, VGG_A

models = ['CPD', 'CPD_A', 'CPD_D19', 'CPD_D19_A', 'CPD_D19_A_avg', 'CPD_D19_A_rfb3', 'CPD_D19_A_rfb3_rfb4', 'CPD_D19_A_rfb3_rfb4_rfb5', 'CPD_D19_A_P']

def load_model(model):
    if model not in models:
        raise ValueError('{} does not exist'.format(model))
    elif model == 'CPD':
        model = CPD()
    elif model == 'CPD_A':
        model = CPD_A()
    elif model == 'CPD_D19':
        model = CPD_D19()
    elif model == 'CPD_D19_A':
        model = CPD_D19_A()
    elif model == 'CPD_D19_A_avg':
        model = CPD_D19_A_avg()
    elif model == 'CPD_D19_A_rfb3':
        model = CPD_D19_A_rfb3()
    elif model == 'CPD_D19_A_rfb3_rfb4':
        model = CPD_D19_A_rfb3_rfb4()
    elif model == 'CPD_D19_A_rfb3_rfb4_rfb5':
        model = CPD_D19_A_rfb3_rfb4_rfb5()
    elif model == 'CPD_D19_A_P':
        model = CPD_D19_A_P()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('{}\t{}'.format(model.name, params))
    return model


class CPD(nn.Module):
    def __init__(self, channel=32):
        super(CPD, self).__init__()
        self.name = 'CPD'
        self.vgg_attention = VGG_A()
        self.vgg_detection = VGG_D()
        self.rfb3_1 = RFB(256, channel)
        self.rfb4_1 = RFB(512, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.rfb3_2 = RFB(256, channel)
        self.rfb4_2 = RFB(512, channel)
        self.rfb5_2 = RFB(512, channel)
        self.agg2 = aggregation(channel)

        self.HA = HA()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4_1, x5_1 = self.vgg_attention(x)
        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4_1)
        x5_1 = self.rfb5_1(x5_1)
        attention = self.agg1(x5_1, x4_1, x3_1)

        x3_2 = self.HA(attention.sigmoid(), x3)

        x4_2, x5_2 = self.vgg_detection(x3_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention), self.upsample(detection)


class CPD_A(nn.Module):
    def __init__(self, channel=32):
        super(CPD_A, self).__init__()
        self.name = 'CPD_A'
        self.vgg = VGG_A()
        self.rfb3_1 = RFB(256, channel)
        self.rfb4_1 = RFB(512, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4, x5 = self.vgg(x)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        x5 = self.rfb5_1(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)


class CPD_D19(nn.Module):
    def __init__(self, channel=32):
        super(CPD_D19, self).__init__()
        self.name = 'CPD_D19'
        self.darknet_attention = D19_A()
        self.darknet_detection = D19_D()
        self.rfb3_1 = RFB(128, channel)
        self.rfb4_1 = RFB(256, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.rfb3_2 = RFB(128, channel)
        self.rfb4_2 = RFB(256, channel)
        self.rfb5_2 = RFB(512, channel)
        self.agg2 = aggregation(channel)

        self.HA = HA()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):

        x3, x4_1, x5_1 = self.darknet_attention(x)
        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4_1)
        x5_1 = self.rfb5_1(x5_1)
        attention = self.agg1(x5_1, x4_1, x3_1)

        x3_2 = self.HA(attention.sigmoid(), x3)

        x4_2, x5_2 = self.darknet_detection(x3_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention), self.upsample(detection)


class CPD_D19_A(nn.Module):
    def __init__(self, channel=32):
        super(CPD_D19_A, self).__init__()
        self.name = 'CPD_D19_A'
        self.darknet = D19_A()
        self.rfb3_1 = RFB(128, channel)
        self.rfb4_1 = RFB(256, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        x5 = self.rfb5_1(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)


class CPD_D19_A_avg(nn.Module):
    def __init__(self, channel=32):
        super(CPD_D19_A_avg, self).__init__()
        self.name = 'CPD_D19_A_avg'
        self.darknet = D19_A_avg()
        self.rfb3_1 = RFB(128, channel)
        self.rfb4_1 = RFB(256, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        x5 = self.rfb5_1(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)


class CPD_D19_A_rfb3(nn.Module):
    def __init__(self, channel=32):
        super(CPD_D19_A_rfb3, self).__init__()
        self.name = 'CPD_D19_A_rfb3'
        self.darknet = D19_A_P()
        self.reduce3 = nn.Conv2d(128, channel, 1)
        self.reduce4 = RFB(256, channel)
        self.rfb5 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x5 = self.rfb5(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)

class CPD_D19_A_rfb3_rfb4(nn.Module):
    def __init__(self, channel=32):
        super(CPD_D19_A_rfb3_rfb4, self).__init__()
        self.name = 'CPD_D19_A_rfb3_rfb4'
        self.darknet = D19_A_P()
        self.reduce3 = nn.Conv2d(128, channel, 1)
        self.reduce4 = nn.Conv2d(256, channel, 1)
        self.rfb5 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x5 = self.rfb5(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)


class CPD_D19_A_rfb3_rfb4_rfb5(nn.Module):
    def __init__(self, channel=32):
        super(CPD_D19_A_rfb3_rfb4_rfb5, self).__init__()
        self.name = 'CPD_D19_A_rfb3_rfb4_rfb5'
        self.darknet = D19_A_P()
        self.reduce3 = nn.Conv2d(128, channel, 1)
        self.reduce4 = nn.Conv2d(256, channel, 1)
        self.rfb5 = RFB_pruned(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x5 = self.rfb5(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)

class CPD_D19_A_P(nn.Module):
    def __init__(self, channel=32):
        super(CPD_D19_A_P, self).__init__()
        self.name = 'CPD_D19_A_P'
        self.darknet = D19_A_P()
        self.reduce3 = nn.Conv2d(128, channel, 1)
        self.reduce4 = nn.Conv2d(256, channel, 1)
        self.rfb5 = RFB_pruned(512, channel)
        self.agg1 = aggregation_pruned(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x5 = self.rfb5(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)
