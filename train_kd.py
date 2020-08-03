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
parser.add_argument('--datasets_path', default='./datasets/train', help='path to datasets, default = ./datasets/train')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--student_model', default='CPD_darknet19_A_minimal', choices=CPD.models, help='chose model, default = CPD')
parser.add_argument('--teacher_model', default='CPD', choices=CPD.models, help='chose model, default = CPD_darknet19_A_minimal')
parser.add_argument('--pth', type=str, default='CPD.pth', help='teacher model filename, default = CPD.pth')
parser.add_argument('--scheduler', type=str, default='CPD_darknet19_A.yaml', help='model filename, default = CPD_darknet19.pth')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs,  default = 100')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate,  default = 0.0001')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size,  default = 32')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin, default = 0.5')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate, default = 0.1')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate,  default = 30')
parser.add_argument('--sensitivity_analysis', action='store_true', default=False)
args = parser.parse_args()

def train(train_loader, model, optimizer, epoch, writer, compression_scheduler=None):
    def add_image(imgs, gts, preds, step, writer):
        writer.add_image('Image', imgs[-1], step)
        writer.add_image('Groundtruth', gts[-1], step)
        writer.add_image('Prediction', preds[-1].sigmoid(), step)

    total_steps = len(train_loader)
    CE = torch.nn.BCEWithLogitsLoss()
    model.train()
    for step, pack in enumerate(train_loader, start=1):
        compression_scheduler.on_minibatch_begin(epoch, step, total_steps, optimizer)
        global_step = (epoch-1) * total_steps + step

        imgs, gts, _, _, _, _ = pack
        imgs = imgs.to(device)
        gts = gts.to(device)
        preds = kd_policy.forward(imgs)
        loss = CE(preds, gts)

        agg_loss = compression_scheduler.before_backward_pass(epoch, step, total_steps, loss, optimizer=optimizer, return_loss_components=True)
        loss = agg_loss.overall_loss
        dist_loss = agg_loss.loss_components[0].value

        writer.add_scalar('Loss', float(loss), global_step)
        writer.add_scalar('Distillation Loss', float(dist_loss), global_step)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        compression_scheduler.before_parameter_optimization(epoch, step, total_steps, optimizer)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch, step, total_steps, optimizer)


        if step == 1 or step % 100 == 0 or step == total_steps:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Distillation Loss: {:.4f}'.
                  format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.epoch, step, total_steps, loss.data, dist_loss.data))
        if step == 1 or step % 500 == 0 or step == total_steps:
            add_image(imgs, gts, preds, global_step, writer)


pruner = args.scheduler.split('.')[1]

device = torch.device(args.device)
print('Device: {}'.format(device))

student = CPD.load_model(args.student_model).to(device)
print('Loaded student:', student.name)
teacher = CPD.load_model(args.teacher_model).to(device)
teacher.load_state_dict(torch.load(args.pth, map_location=torch.device(device)))
print('Loaded teacher:', teacher.name)
optimizer = torch.optim.Adam(student.parameters(), args.lr)

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

save_dir = os.path.join('pruning', student.name, 'kd')

dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
writer = tensorboard.SummaryWriter(os.path.join(save_dir, 'logs'))
print('Dataset loaded successfully')

compression_scheduler = distiller.file_config(student, optimizer, args.scheduler, None, None)
dlw = distiller.DistillationLossWeights(0.5, 0.5,0)
kd_policy = distiller.KnowledgeDistillationPolicy(student, teacher, 1, dlw)
compression_scheduler.add_policy(kd_policy, starting_epoch=1, ending_epoch=100, frequency=1)
for epoch in range(1, args.epoch+1):

    compression_scheduler.on_epoch_begin(epoch)
    print('Started epoch {:03d}/{:03d}'.format(epoch, args.epoch))
    writer.add_scalar('Learning rate', float(optimizer.param_groups[0]['lr']), epoch)
    train(train_loader, student, optimizer, epoch, writer, compression_scheduler)
    compression_scheduler.on_epoch_end(epoch)
    torch.save(student.state_dict(), '{}/{}.kd.pth'.format(save_dir, student.name))
