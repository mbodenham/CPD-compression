#ssh -L 16006:127.0.0.1:16006 mb2775@ogg.cs.bath.ac.uk
import torch
import torchvision.transforms as transforms
import torch.utils.tensorboard  as tensorboard

import distiller

import os, argparse
from datetime import datetime
from functools import partial
import numpy as np

import CPD

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='./datasets/test_small', help='path to datasets, default = ./datasets/train')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--model', default='CPD_D19_A', choices=CPD.models, help='chose model, default = CPD_darknet19')
parser.add_argument('--pth', type=str, default='ckpts/CPD_D19_A.pth', help='model filename, default = CPD_darknet19.pth')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
args = parser.parse_args()


def test(test_loader, model, criterion, loggers=None, activations_collectors=None, args=None):
    s = np.zeros(len(test_loader))
    losses = np.zeros(len(test_loader))
    model.eval()
    eval = CPD.Eval('./datasets/test_small/', model.name)
    with torch.no_grad():
        for step, pack in enumerate(test_loader):
            imgs, gts, _, _, _, _ = pack
            imgs = imgs.to(device)
            gts = gts.to(device)
            if '_A' in model.name:
                preds = model(imgs)
                loss = criterion(preds, gts)
            s[step] = eval.smeasure(preds.sigmoid(), gts,'test')['test']
            losses[step] = loss
    return s.mean(), s.mean(), losses.mean()

device = torch.device(args.device)
print('Device: {}'.format(device))

model = CPD.load_model(args.model).to(device)
model.load_state_dict(torch.load(args.pth, map_location=torch.device(device)))

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
test_fnc = partial(test, test_loader=test_loader, criterion=torch.nn.BCEWithLogitsLoss())
params = [['darknet.conv1.conv1_1.weight',
          'darknet.conv2.conv2_1.weight',
          'darknet.conv3.conv3_1.weight',
          'darknet.conv3.conv3_2.weight',
          'darknet.conv3.conv3_3.weight',
          'darknet.conv4.conv4_1.weight',
          'darknet.conv4.conv4_2.weight',
          'darknet.conv4.conv4_3.weight',
          'darknet.conv5.conv5_1.weight',
          'darknet.conv5.conv5_2.weight',
          'darknet.conv5.conv5_3.weight'],
         ['rfb3_1.branch0.0.weight',
          'rfb3_1.branch1.0.weight',
          'rfb3_1.branch1.1.weight',
          'rfb3_1.branch1.2.weight',
          'rfb3_1.branch1.3.weight',
          'rfb3_1.branch2.0.weight',
          'rfb3_1.branch2.1.weight',
          'rfb3_1.branch2.2.weight',
          'rfb3_1.branch2.3.weight',
          'rfb3_1.branch3.0.weight',
          'rfb3_1.branch3.1.weight',
          'rfb3_1.branch3.2.weight',
          'rfb3_1.branch3.3.weight',
          'rfb3_1.conv_cat.weight',
          'rfb3_1.conv_res.weight'],
         ['rfb4_1.branch0.0.weight',
          'rfb4_1.branch1.0.weight',
          'rfb4_1.branch1.1.weight',
          'rfb4_1.branch1.2.weight',
          'rfb4_1.branch1.3.weight',
          'rfb4_1.branch2.0.weight',
          'rfb4_1.branch2.1.weight',
          'rfb4_1.branch2.2.weight',
          'rfb4_1.branch2.3.weight',
          'rfb4_1.branch3.0.weight',
          'rfb4_1.branch3.1.weight',
          'rfb4_1.branch3.2.weight',
          'rfb4_1.branch3.3.weight',
          'rfb4_1.conv_cat.weight',
          'rfb4_1.conv_res.weight'],
         ['rfb5_1.branch0.0.weight',
          'rfb5_1.branch1.0.weight',
          'rfb5_1.branch1.1.weight',
          'rfb5_1.branch1.2.weight',
          'rfb5_1.branch1.3.weight',
          'rfb5_1.branch2.0.weight',
          'rfb5_1.branch2.1.weight',
          'rfb5_1.branch2.2.weight',
          'rfb5_1.branch2.3.weight',
          'rfb5_1.branch3.0.weight',
          'rfb5_1.branch3.1.weight',
          'rfb5_1.branch3.2.weight',
          'rfb5_1.branch3.3.weight',
          'rfb5_1.conv_cat.weight',
          'rfb5_1.conv_res.weight'],
         ['agg1.conv_upsample1.weight',
          'agg1.conv_upsample2.weight',
          'agg1.conv_upsample3.weight',
          'agg1.conv_upsample4.weight',
          'agg1.conv_concat2.weight',
          'agg1.conv_upsample5.weight',
          'agg1.conv_concat3.weight',
          'agg1.conv4.weight',
          'agg1.conv5.weight']]
          
for params, fname in zip(params, ['darknet', 'rfb3_1', 'rfb4_1', 'rfb5_1', 'agg1']):
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=params,
                                                         sparsities=np.arange(0,1,0.05),
                                                         test_func=test_fnc,
                                                         group='filter')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity_{}.csv'.format(fname))
print('Complete')
