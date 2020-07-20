import argparse
import csv
import torch
from CPD import CPD, CPD_darknet19, ImageGroundTruthFolder
from torchvision import transforms
from os import walk
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='./datasets/prune_test', help='path to datasets, default = ./datasets/test')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
args = parser.parse_args()

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

f = []
for (dirpath, dirnames, filenames) in walk('pruned/'):
    f.extend(filenames)
    break
print(f)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = CPD().to(device)

dataset = ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
datasets = [d.name for d in os.scandir(args.datasets_path) if d.is_dir()]
models = []

for pth in f:

    state_dict = torch.load('pruned/' + pth, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    print('Loaded:', pth)
    mae = {ds_name: [] for ds_name in datasets}

    for idx, pack in enumerate(loader):
        img, gt, dataset, img_name, img_res = pack
        gt = gt.to(device)
        if idx % 100 == 0 or idx == len(loader):
            print('[{}/{}] {} - {}'.format(idx, len(loader), dataset[0], img_name[0]))
        img = img.to(device)
        _, pred = model(img)
        mae[datasets[0]].append(torch.abs(pred.sigmoid() - gt).mean().cpu().detach().numpy())

    for d in datasets:
        models.append([pth, np.mean(mae[d])])
    print(models)

for m in models:
    print(m)
