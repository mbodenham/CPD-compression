import torch
import torch.nn as nn

from .darknet import Darknet19, Darknet19_A, Darknet19_A_pruned
from .evaluate import Eval
from .dataset import EvalImageGroundTruthFolder, ImageGroundTruthFolder
from .modules import aggregation, HA, RFB, aggregation_minimal, RFB_minimal
from .vgg import B2_VGG

models = ['CPD', 'CPD_darknet19', 'CPD_darknet19_A', 'CPD_darknet19_A_pruned', 'CPD_darknet19_A_minimal']

def load_model(model):
    if model not in models:
        raise ValueError('{} does not exist'.format(model))
    elif model == 'CPD':
        model = CPD()
    elif model == 'CPD_darknet19':
        model = CPD_darknet19()
    elif model == 'CPD_darknet19_A':
        model = CPD_darknet19_A()
    elif model == 'CPD_darknet19_A_pruned':
        model = CPD_darknet19_A_pruned()
    elif model == 'CPD_darknet19_A_minimal':
        model = CPD_darknet19_A_minimal()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('{}\t{}'.format(model.name, params))
    return model

class CPD(nn.Module):
    def __init__(self, channel=32):
        super(CPD, self).__init__()
        self.name = 'CPD'
        self.vgg = B2_VGG()
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
        modules = [self.vgg, self.rfb3_1, self.rfb4_1, self.rfb5_1, self.agg1,
                   self.rfb3_2, self.rfb4_2, self.rfb5_2, self.agg2, self.HA, self.upsample]
        modules_names = ['vgg', 'rfb3_1', 'rfb4_1', 'rfb5_1', 'agg1',
                   'rfb3_2', 'rfb4_2', 'rfb5_2', 'agg2', 'HA', 'upsample']
        print('Parameters')
        for module, name in zip(modules, modules_names):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print('{}\t{}'.format(name, params))

    def forward(self, x):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)

        x3_1 = x3
        x4_1 = self.vgg.conv4_1(x3_1)
        x5_1 = self.vgg.conv5_1(x4_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        x5_1 = self.rfb5_1(x5_1)
        attention = self.agg1(x5_1, x4_1, x3_1)

        x3_2 = self.HA(attention.sigmoid(), x3)
        x4_2 = self.vgg.conv4_2(x3_2)
        x5_2 = self.vgg.conv5_2(x4_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention), self.upsample(detection)


class CPD_darknet19(nn.Module):
    def __init__(self, channel=32):
        super(CPD_darknet19, self).__init__()
        self.name = 'CPD_darknet19'
        self.darknet = Darknet19()
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
        modules = [self.darknet, self.rfb3_1, self.rfb4_1, self.rfb5_1, self.agg1,
                   self.rfb3_2, self.rfb4_2, self.rfb5_2, self.agg2, self.HA, self.upsample]
        modules_names = ['darknet', 'rfb3_1', 'rfb4_1', 'rfb5_1', 'agg1',
                   'rfb3_2', 'rfb4_2', 'rfb5_2', 'agg2', 'HA', 'upsample']
        print('Parameters')
        for module, name in zip(modules, modules_names):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print('{}\t{}'.format(name, params))

    def forward(self, x):

        x3, x4_1, x5_1, x4_1, x5_1 = self.darknet(x)

        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4_1)
        x5_1 = self.rfb5_1(x5_1)
        attention = self.agg1(x5_1, x4_1, x3_1)

        x3_2 = self.HA(attention.sigmoid(), x3)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention), self.upsample(detection)


class CPD_darknet19_A(nn.Module):
    def __init__(self, channel=32):
        super(CPD_darknet19_A, self).__init__()
        self.name = 'CPD_darknet19_A'
        self.darknet = Darknet19_A()
        self.rfb3_1 = RFB(128, channel)
        self.rfb4_1 = RFB(256, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        modules = [self.darknet, self.rfb3_1, self.rfb4_1, self.rfb5_1, self.agg1, self.upsample]
        modules_names = ['darknet', 'rfb3_1', 'rfb4_1', 'rfb5_1', 'agg1', 'upsample']
        print('Parameters')
        for module, name in zip(modules, modules_names):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print('{}\t{}'.format(name, params))

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        x5 = self.rfb5_1(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)

class CPD_darknet19_A_pruned(nn.Module):
    def __init__(self, channel=32):
        super(CPD_darknet19_A_pruned, self).__init__()
        self.name = 'CPD_darknet19_A_pruned'
        self.darknet = Darknet19_A_pruned()
        self.rfb3_1 = RFB(128, channel)
        self.rfb4_1 = RFB(256, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        modules = [self.darknet, self.rfb3_1, self.rfb4_1, self.rfb5_1, self.agg1, self.upsample]
        modules_names = ['darknet', 'rfb3_1', 'rfb4_1', 'rfb5_1', 'agg1', 'upsample']
        print('Parameters')
        for module, name in zip(modules, modules_names):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print('{}\t{}'.format(name, params))

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        x5 = self.rfb5_1(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)

class CPD_darknet19_A_minimal(nn.Module):
    def __init__(self, channel=32):
        super(CPD_darknet19_A_minimal, self).__init__()
        self.name = 'CPD_darknet19_A_minimal'
        self.darknet = Darknet19_A_pruned()
        self.reduce3 = nn.Conv2d(128, channel, 1)
        self.reduce4 = nn.Conv2d(256, channel, 1)
        self.rfb5 = RFB_minimal(512, channel)
        self.agg1 = aggregation_minimal(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        #modules = [self.darknet, self.rfb3_1, self.rfb4_1, self.rfb5_1, self.agg1, self.upsample]
        #modules_names = ['darknet', 'rfb3_1', 'rfb4_1', 'rfb5_1', 'agg1', 'upsample']
        #print('Parameters')
        #for module, name in zip(modules, modules_names):
        #    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        #    print('{}\t{}'.format(name, params))

    def forward(self, x):
        x3, x4, x5 = self.darknet(x)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x5 = self.rfb5(x5)
        attention = self.agg1(x5, x4, x3)

        return self.upsample(attention)
