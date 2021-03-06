import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as utils
import torchvision.transforms as transforms

from tqdm import tqdm
import time
import numpy as np
import pdb, os, argparse

import CPD

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, default='./datasets/test', help='path to datasets, default = ./datasets/test')
parser.add_argument('--save_path', type=str, default=False, help='path to save results, default = ./results')
parser.add_argument('--model', default='CPD_darknet19', choices=CPD.models, help='chose model, default = CPD_darknet19')
parser.add_argument('--pth', type=str, default='CPD_darknet19.pth', help='model filename, default = CPD_darknet19.pth')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
parser.add_argument('--reps', type=int, default=10000, help='image input and output resolution, default = 352')
parser.add_argument('--time', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
args = parser.parse_args()

device = torch.device(args.device)
print('Device: {}'.format(device))

model = CPD.load_model(args.model).to(device)
print('Loaded:', model.name)

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

def get_pred(model, input):
    if '_A' in model.name:
        pred = model(input)
    else:
        _, pred = model(input)

    return pred

model.eval()
with torch.no_grad():
    if args.time:
        input = torch.rand([1, 3, args.imgres, args.imgres]).to(device)
        times = np.zeros(args.reps)

        t0 = time.time()
        while time.time() - t0 < 60:
            get_pred(model, input)

        with torch.autograd.profiler.profile() as prof:
            get_pred(model, input)

        tt = time.time()
        idx = 0
        while time.time() - tt < 180:
            t0 = time.time()
            get_pred(model, input)
            times[idx] = time.time() - t0
            idx += 1

        avg_t = np.mean(times[np.nonzero(times)])
        print('Inference time', avg_t)
        print('Std', np.std(times[np.nonzero(times)]))
        print('FPS', 1/avg_t)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    if args.eval:
        model.load_state_dict(torch.load(args.pth, map_location=torch.device(device)))
        dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        eval = CPD.Eval(args.datasets_path, model.name)
        eval.to(device)

        for pack in tqdm(test_loader):
            img, gt, dataset, img_name, _, _ = pack
            img = img.to(device)
            gt = gt.to(device)

            pred = get_pred(model, img)
            eval.run(pred.sigmoid(), gt, dataset)

        results = eval.results()
        for dataset in results.keys():
            print(dataset)
            for metric in results[dataset].keys():
                print('\t{}\t{:.4f}'.format(metric, results[dataset][metric]))
