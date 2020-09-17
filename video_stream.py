import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import CPD

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='CPD_D19_A_P', choices=CPD.models, help='chose model, default = CPD_D19_A_P')
parser.add_argument('--pth', type=str, default='ckpts/CPD_D19_A_P.pth', help='model filename, default = CPD_D19_A_P.pth')
parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--imgres', type=int, default=224, help='image input and output resolution, default = 224')
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


cap = cv2.VideoCapture(0)
fps = 25
resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

out = cv2.VideoWriter('./videos/stream_out.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
sbs = cv2.VideoWriter('./videos/stream_sbs_out.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (resolution[0], resolution[1]//2))
while True:
    _, frame = cap.read()

    frame_in = torch.unsqueeze(transform(frame), 0)
    frame_pred = model(frame_in)
    frame_out = frame_pred.sigmoid().data.cpu()[0]
    frame_out = np.array(transforms.ToPILImage()(frame_out).convert("L")).astype(np.uint8)

    frame_out_pp = frame_out.copy()
    frame_out_pp = cv2.erode(frame_out_pp, np.ones((5,5), np.uint8))
    frame_out_pp = cv2.erode(frame_out_pp, np.ones((10,10), np.uint8))
    frame_out_pp = cv2.dilate(frame_out_pp, np.ones((10,10), np.uint8))
    frame_out_pp = cv2.dilate(frame_out_pp, np.ones((5,5), np.uint8), iterations=2)
    frame_out_pp = cv2.blur(frame_out_pp, (20,20))
    frame_out_pp = cv2.resize(frame_out_pp, resolution)

    frame_out_pp = cv2.cvtColor(frame_out_pp, cv2.COLOR_GRAY2BGR)
    out.write(frame_out_pp)

    frame_out_pp_pp = frame_out_pp.copy()
    frame_out_pp = np.concatenate((frame, frame_out_pp), axis=1)
    frame_out_pp = cv2.resize(frame_out_pp, (resolution[0], resolution[1]//2))
    sbs.write(frame_out_pp)

    cv2.imshow('final', frame_out_pp)
    cv2.waitKey(1)

out.release()
sbs.release()
cap.release()
cv2.destroyAllWindows()
