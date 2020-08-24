import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import time
import numpy as np
import os, argparse
from collections import deque
import CPD

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='v33_Hugo_87461', help='path to datasets, default = ./datasets/test')
parser.add_argument('--model', default='CPD_darknet19_A_minimal', choices=CPD.models, help='chose model, default = CPD_darknet19')
parser.add_argument('--pth', type=str, default='CPD_darknet19_A_minimal.pth', help='model filename, default = CPD_darknet19.pth')
parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
args = parser.parse_args()

device = torch.device(args.device)
print('Device: {}'.format(device))
model = CPD.load_model(args.model).to(device)
model.load_state_dict(torch.load(args.pth, map_location=torch.device(device)))
model.eval()
print('Loaded:', model.name)

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_r = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

video_name = args.video
video_dir = './videos/'+video_name+'/'

cap = cv2.VideoCapture(video_dir+'source.avi')
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

out = cv2.VideoWriter(video_dir+'saliency_map.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (resolution[0], resolution[1]))
out_sbs = cv2.VideoWriter(video_dir+'sbs.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (resolution[0], resolution[1]//2))
prev_frame = deque(maxlen=3)
for idx in range(nframes):
    print(video_name,idx,'/',nframes)
    _, frame = cap.read()

    # frame_o = torch.Tensor(frame)
    frame_in = torch.unsqueeze(transform(frame), 0)
    frame_pred = model(frame_in)
    frame_out = F.interpolate(frame_pred, size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)
    frame_out = frame_out.sigmoid().data.cpu()[0]
    frame_out = np.array(transforms.ToPILImage()(frame_out).convert("L")).astype(np.uint8)

    prev_frame.append(frame_out)
    if len(prev_frame) == 3:
        frame_out_pp = np.average(np.array(prev_frame), axis=0, weights=[1, 2, 5]).astype(np.uint8)
    else:
        frame_out_pp = np.mean(np.array(prev_frame), axis=0).astype(np.uint8)

    # frame_out_pp = frame_out.copy()
    frame_out_pp[frame_out_pp>=0.5] = 1
    frame_out_pp[frame_out_pp<0.5] = 0
    frame_out_pp = cv2.dilate(frame_out, np.ones((10,10), np.uint8) , iterations=10)
    frame_out_pp = cv2.blur(frame_out_pp, (100,100))


    frame_out_pp = cv2.cvtColor(frame_out_pp, cv2.COLOR_GRAY2BGR)
    out.write(frame_out_pp)
    frame_out_pp = np.concatenate((frame, frame_out_pp), axis=1)
    frame_out_pp = cv2.resize(frame_out_pp, (resolution[0], resolution[1]//2))

    out_sbs.write(frame_out_pp)

#out.release()
out_sbs.release()
cap.release()
cv2.destroyAllWindows()
