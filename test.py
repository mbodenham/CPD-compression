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
parser.add_argument('--reps', type=int, default=100, help='image input and output resolution, default = 352')
parser.add_argument('--time', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
args = parser.parse_args()

device = torch.device(args.device)
print('Device: {}'.format(device))

model = CPD.load_model(args.model).to(device)

model.load_state_dict(torch.load(args.pth, map_location=torch.device(device)))
model.eval()
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

if args.time:
    model.eval()
    with torch.no_grad():
        n = 100
        input = torch.rand([1, 3, args.imgres, args.imgres]).to(device)
        times = np.zeros(args.reps)

        for warm_up in range(args.reps//2):
            get_pred(model, input)


        with torch.autograd.profiler.profile() as prof:
            get_pred(model, input)

        for rep in range(args.reps):
            t0 = time.time()
            get_pred(model, input)
            times[rep] = time.time() - t0

        avg_t = np.mean(times)
    print('Inference time', avg_t)
    print('Std', np.std(times))
    print('FPS', 1/avg_t)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

elif args.eval:
    dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
<<<<<<< HEAD
    eval = CPD.Eval(args.datasets_path, model.name)
    eval.to(device)
=======
    eval = CPD.Eval(args.datasets_path, model.name).to(device)
>>>>>>> 0701d14e71ffc87788cd1a00b990544f6382d199

    for pack in tqdm(test_loader):
        img, gt, dataset, img_name, _, _ = pack
        #print('[{:.2f}%] {} - {}'.format((idx/len(test_loader))*100, dataset[0], img_name[0]))
        img = img.to(device)
        gt = gt.to(device)

        pred = get_pred(model, img)
        eval.run(pred.sigmoid(), gt, dataset)

    print(eval.results())
