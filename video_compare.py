import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import time
import numpy as np
import os, argparse
from collections import deque
import CPD
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='hike', help='video to load, default = hike')
parser.add_argument('--bitrate', type=int, default=1000, help='image input and output resolution, default = 352')
args = parser.parse_args()

video_name = args.video
video_dir = './videos/'+video_name+'/'
bitrate = args.bitrate
bitrate_dir =  str(args.bitrate) + '/'

cap_gt = cv2.VideoCapture(video_dir+'gt.avi')
cap_src = cv2.VideoCapture(video_dir+'source.avi')
cap_std = cv2.VideoCapture(video_dir+bitrate_dir+'standard_compression.mkv')
cap_sal = cv2.VideoCapture(video_dir+bitrate_dir+'salient_compression.mkv')
cap_sal_cpd = cv2.VideoCapture(video_dir+bitrate_dir+'salient_compression_cpd.mkv')
cap_gt_sal = cv2.VideoCapture(video_dir+bitrate_dir+'gt_compression.mkv')

nframes = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap_src.get(cv2.CAP_PROP_FPS))
resolution = (int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT)))

ewssim_std = np.zeros(nframes)
ewssim_sal = np.zeros(nframes)
ewssim_sal_cpd = np.zeros(nframes)
ewssim_gt_sal = np.zeros(nframes)

for idx in tqdm(range(nframes), desc='{} - {}'.format(video_name, bitrate)):
    _, frame_gt = cap_gt.read()
    frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
    _, frame_src = cap_src.read()
    _, frame_std = cap_std.read()
    _, frame_sal = cap_sal.read()
    _, frame_sal_cpd = cap_sal_cpd.read()
    _, frame_gt_sal = cap_gt_sal.read()


    s_std = ssim(frame_src, frame_std, multichannel=True, full=True)
    ewssim_std[idx] = (frame_gt * s_std[1].mean(axis=2)).sum() / frame_gt.sum()
    s_sal = ssim(frame_src, frame_sal, multichannel=True, full=True)
    ewssim_sal[idx] = (frame_gt * s_sal[1].mean(axis=2)).sum() / frame_gt.sum()
    s_gt_sal = ssim(frame_src, frame_gt_sal, multichannel=True, full=True)
    ewssim_gt_sal[idx] = (frame_gt * s_gt_sal[1].mean(axis=2)).sum() / frame_gt.sum()
    s_sal_cpd = ssim(frame_src, frame_sal_cpd, multichannel=True, full=True)
    ewssim_sal_cpd[idx] = (frame_gt * s_sal_cpd[1].mean(axis=2)).sum() / frame_gt.sum()

print('std', np.nanmean(ewssim_std))
print('sal', np.nanmean(ewssim_sal))
print('gt_sal', np.nanmean(ewssim_gt_sal))
print('cpd_sal', np.nanmean(ewssim_sal_cpd))
print()
cap_gt.release()
cap_src.release()
cap_std.release()
cap_sal.release()
cap_sal_cpd.release()

with open('{}{}.csv'.format(video_dir, 'results'), 'a+', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([args.bitrate, np.nanmean(ewssim_std), np.nanmean(ewssim_sal), np.nanmean(ewssim_gt_sal), np.nanmean(ewssim_sal_cpd)])
