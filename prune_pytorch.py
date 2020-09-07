import torch
import torchvision.transforms as transforms
import torch.utils.tensorboard  as tensorboard
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

import os, argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import CPD

def get_pred(model, input):
    if '_A' in model.name:
        pred = model(input)
    else:
        _, pred = model(input)

    return pred
    
parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='./datasets/val', help='path to datasets, default = ./datasets/test')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
args = parser.parse_args()

device = torch.device('cpu')
state_dict = torch.load('ckpts/CPD_D19_A_avg.pth', map_location=torch.device(device))
model = CPD.load_model('CPD_D19_A_avg').to(device)

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])
dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)

for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    model.load_state_dict(state_dict)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append([module, 'weight'])

    prune.global_unstructured(parameters_to_prune,
                              pruning_method=prune.L1Unstructured,
                              amount=i,)


    nelements = 0
    weight_sum = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            nelements += module.weight.nelement()
            weight_sum += torch.sum(module.weight == 0)

    gs = 100 *  weight_sum // nelements
    for para in parameters_to_prune:
        prune.remove(para[0], para[1])

    with torch.no_grad():
        model.eval()
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        eval = CPD.Eval(args.datasets_path, model.name)
        eval.to(device)
        s = np.zeros(len(test_loader))

        for idx, pack in enumerate(tqdm(test_loader)):
            img, gt, dataset, img_name, _, _ = pack
            img = img.to(device)
            gt = gt.to(device)

            pred = get_pred(model, img)
            s[idx] = eval.smeasure_only(pred.sigmoid(), gt)

        model.train()
        print(gs, s.mean())
