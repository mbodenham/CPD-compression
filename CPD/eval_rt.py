## https://github.com/Hanqer/Evaluate-SOD/blob/master/evaluator.py

import csv
import os
import time

import numpy as np
import torch
from torchvision import transforms


class Eval():
    def __init__(self, dataset_dir, model_name):
        self.model_name = model_name
        self.datasets = [d.name for d in os.scandir(dataset_dir) if d.is_dir()]
        print('Datasets', self.datasets)
        self.mae = {ds_name: [] for ds_name in self.datasets}
        self.avgF = {ds_name: [] for ds_name in self.datasets}
        self.maxF = {ds_name: [] for ds_name in self.datasets}
        self.S = {ds_name: [] for ds_name in self.datasets}
        self.metrics = {ds_name: {} for ds_name in self.datasets}
        print(self.metrics)
        print(self.datasets)

    def run(self, pred, gt, dataset):
        self.MAE(pred, gt, dataset)
        self.fmeasure(pred, gt, dataset)
        self.smeasure(pred, gt, dataset)

    def results(self):
        for dataset in self.datasets:
            self.metrics[dataset]['MAE'] = np.mean(self.mae[dataset])
            self.metrics[dataset]['avgF'] = np.mean(self.avgF[dataset])
            self.metrics[dataset]['maxF'] = np.mean(self.maxF[dataset])
            self.metrics[dataset]['S'] = np.nanmean(self.S[dataset])

        if print:
            header = []
            for dataset in self.datasets:
                header.append(dataset)
                header.append('')
                header.append('')
                header.append('')
            metrics = ['MAE', 'avgF', 'maxF', 'S'] * 4
            results = []
            for dataset in self.metrics.values():
                for result in dataset.values():
                    results.append(result)

            filename = 'metrics_{}.csv'.format(self.model_name)
            with open(filename, 'w') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerow(metrics)
                writer.writerow(results)
        return self.metrics

    def MAE(self, pred, gt, dataset):
        with torch.no_grad():
            self.mae[dataset[0]].append(torch.abs(pred - gt).mean().cpu().numpy())

    def fmeasure(self, pred, gt, dataset):
        beta2 = 0.3
        with torch.no_grad():
            prec, recall = self._eval_pr(pred, gt, 255)
            f_score = ((1 + beta2) * prec * recall) / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0 # for Nan
            self.avgF[dataset[0]].append(torch.mean(f_score).cpu().numpy())
            self.maxF[dataset[0]].append(self._eval_fmax(prec, recall, beta2).cpu())


    def smeasure(self, pred, gt, dataset):
        alpha = 0.5

        with torch.no_grad():
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                gt[gt>=0.5] = 1
                gt[gt<0.5] = 0
                Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])

            self.S[dataset[0]].append(Q.cpu().numpy())

    def _eval_pr(self, y_pred, y, num):
        if torch.cuda.is_available():
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _eval_fmax(self, prec, recall, beta2):
        idx = torch.argmax(prec + recall)
        f_score = ((1 + beta2) * prec[idx] * recall[idx]) / (beta2 * prec[idx] + recall[idx])
        f_score[f_score != f_score] = 0
        return f_score

    def _S_object(self, pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if torch.cuda.is_available():
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if torch.cuda.is_available():
                i = torch.from_numpy(np.arange(0,cols)).cuda().float()
                j = torch.from_numpy(np.arange(0,rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0,cols)).float()
                j = torch.from_numpy(np.arange(0,rows)).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q
