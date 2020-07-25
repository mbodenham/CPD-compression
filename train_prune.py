#ssh -L 16006:127.0.0.1:16006 mb2775@ogg.cs.bath.ac.uk
import torch
import torchvision.transforms as transforms
import torch.utils.tensorboard  as tensorboard

import distiller

import os, argparse
from datetime import datetime

import CPD

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default='./datasets/train', help='path to datasets, default = ./datasets/train')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--model', default='CPD_darknet19_A', choices=CPD.models, help='chose model, default = CPD_darknet19')
parser.add_argument('--pth', type=str, default='CPD_darknet19_A.pth', help='model filename, default = CPD_darknet19.pth')
parser.add_argument('--scheduler', type=str, default='CPD_darknet19_A.yaml', help='model filename, default = CPD_darknet19.pth')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs,  default = 100')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate,  default = 0.0001')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size,  default = 10')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin, default = 0.5')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate, default = 0.1')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate,  default = 30')
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
        optimizer.zero_grad()
        imgs, gts, _, _, _, _ = pack
        imgs = imgs.to(device)
        gts = gts.to(device)
        if '_A' in model.name:
            preds = model(imgs)
            loss = CE(preds, gts)
            writer.add_scalar('Loss', float(loss), global_step)
        else:
            atts, preds = model(imgs)
            att_loss = CE(atts, gts)
            det_loss = CE(preds, gts)
            loss = att_loss + det_loss
            writer.add_scalar('Loss/Attention Loss', float(att_loss), global_step)
            writer.add_scalar('Loss/Detection Loss', float(det_loss),global_step)
            writer.add_scalar('Loss/Total Loss', float(loss), global_step)
        loss = compression_scheduler.before_backward_pass(epoch, train_step, total_steps, loss,
                                                                optimizer=optimizer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        compression_scheduler.before_parameter_optimization(epoch, train_step, total_steps, optimizer)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch, train_step, total_steps, optimizer)


        if step % 100 == 0 or step == total_steps:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.epoch, step, total_steps, loss.data))
        if step == 1 or step % 500 == 0 or step == total_steps:
            add_image(imgs, gts, preds, global_step, writer)

    save_path = 'ckpts/{}/'.format(model.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if epoch % 5 == 0:
    #     torch.save(model.state_dict(), '{}{}.pth.{:03d}'.format(save_path, model.name, epoch))
    torch.save(model.state_dict(), '{}{}.pth'.format(save_path, model.name))

device = torch.device(args.device)
print('Device: {}'.format(device))

model = CPD.load_model(args.model).to(device)

optimizer = torch.optim.Adam(model.parameters(), args.lr)

compression_scheduler = distiller.file_config(model, optimizer, args.scheduler, None, None)

model.load_state_dict(torch.load(args.pth, map_location=torch.device(device)))

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])

dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
writer = tensorboard.SummaryWriter(os.path.join('logs', model.name, datetime.now().strftime('%Y%m%d-%H%M%S')))
print('Dataset loaded successfully')
for epoch in range(1, args.epoch+1):
    compression_scheduler.on_epoch_begin(epoch)
    print('Started epoch {:03d}/{}'.format(epoch, args.epoch))
    lr_lambda = lambda epoch: args.decay_rate ** (epoch // args.decay_epoch)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    train(train_loader, model, optimizer, epoch, writer, compression_scheduler)
