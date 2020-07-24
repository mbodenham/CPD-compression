import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as utils
import torchvision.transforms as transforms

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

if args.eval:
    eval = CPD.Eval(args.datasets_path, model.name)

if args.time:
    model.eval()
    with torch.no_grad():
        n = 100
        input = torch.rand([1, 3, args.imgres, args.imgres]).to(device)
        times = np.zeros(args.reps)

        with torch.autograd.profiler.profile() as prof:
            if '_A' in model.name:
                pred = model(input)
            else:
                _, pred = model(input)

        for rep in range(args.reps):
            t0 = time.time()
            if '_A' in model.name:
                pred = model(input)
            else:
                _, pred = model(input)
            times[rep] = time.time() - t0

        avg_t = np.mean(times)
    print('Inference time', avg_t)
    print('Std', np.std(times))
    print('FPS', 1/avg_t)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

else:
    dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, pack in enumerate(test_loader):
        img, gt, dataset, img_name, img_res, orig = pack
        print('{} - {}'.format(dataset[0], img_name[0]))
        img = img.to(device)

        if '_A' in model.name:
            pred = model(img)
        else:
            _, pred = model(img)

        if args.eval:
            gt = gt.to(device)
            eval.run(pred.sigmoid(), gt, dataset)

        if args.save_path and idx % 100 == 0:
            save_path = './results/{}/{}/'.format(model.name, dataset[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            grid_img = utils.make_grid(torch.cat((orig.cpu(), gt.cpu().repeat(1, 3, 1, 1), pred.sigmoid().cpu().repeat(1, 3, 1, 1))))
            filename = '{}grid_{}.png'.format(save_path, img_name[0])
            utils.save_image(grid_img,  filename)

            # pred = F.interpolate(pred, size=img_res[::-1], mode='bilinear', align_corners=False)
            # pred = pred.sigmoid().data.cpu()
            #
            # filename = '{}{}.png'.format(save_path, img_name[0])
            # utils.save_image(pred,  filename)

    if args.eval:
        print(eval.results())
