import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.utils.tensorboard  as tensorboard

import os, argparse
from datetime import datetime
import numpy as np

import CPD

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default='./datasets/train', help='path to datasets, default = ./datasets/train')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu, default = cuda')
parser.add_argument('--model', default='CPD', choices=CPD.models, help='chose model, default = CPD')
parser.add_argument('--imgres', type=int, default=352, help='image input and output resolution, default = 352')
parser.add_argument('--crop_imgres', type=int, default=352, help='image input and output resolution, default = 352')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs,  default = 100')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate,  default = 0.0001')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size,  default = 10')
parser.add_argument('--lr_patience', type=int, default=2, help='lr decay after n epochs, default = 2')
parser.add_argument('--training_patience', type=int, default=6, help='stop training after n epochs, default = 4')
parser.add_argument('--rand_crop', action='store_true', default=False)
args = parser.parse_args()

def train(model, train_loader, optimizer, epoch, writer):
    total_steps = len(train_loader)
    CE = torch.nn.BCEWithLogitsLoss()
    model.train()
    for step, pack in enumerate(train_loader, start=1):
        global_step = (epoch-1) * total_steps + step
        optimizer.zero_grad()
        imgs, gts, _, _, _, origs = pack
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
            writer.add_scalar('Loss', float(det_loss), global_step)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == total_steps:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.epoch, step, total_steps, loss.data))
        if step == 1 or step == total_steps//2 or step == total_steps:
            img = utils.make_grid(torch.cat((torch.unsqueeze(origs[-1].cpu(), 0), torch.unsqueeze(gts[-1].cpu().repeat(3, 1, 1), 0), torch.unsqueeze(preds[-1].sigmoid().cpu().repeat(3, 1, 1), 0))))
            writer.add_image('Prediction', img, global_step)


def validate(model, val_loader, val_writer, epoch, global_step):
    model.eval()
    eval = CPD.Eval('./datasets/val', model.name)
    eval.to(device)
    with torch.no_grad():

        s = np.zeros(len(val_loader))
        mae = s.copy()
        val_loss = s.copy()
        for idx, pack in enumerate(val_loader):
            img, gt, _, _, _, orig = pack
            img = img.to(device)
            gt = gt.to(device)

            if '_A' in model.name:
                pred = model(img)
            else:
                _, pred = model(img)
            s[idx] = eval.smeasure_only(pred.sigmoid(), gt)
            mae[idx] = torch.nn.L1Loss()(pred.sigmoid(), gt)
            val_loss[idx] = torch.nn.BCEWithLogitsLoss()(pred, gt)

    model.train()
    val_writer.add_scalar('Metrics/S-Measure', float(s.mean()), global_step)
    val_writer.add_scalar('Metrics/MAE', float(mae.mean()), global_step)
    val_writer.add_scalar('Loss', float(val_loss.mean()), global_step)
    img = utils.make_grid(torch.cat((orig.cpu(), gt.cpu().repeat(1, 3, 1, 1), pred.sigmoid().cpu().repeat(1, 3, 1, 1))))
    val_writer.add_image('Prediction', img, global_step)
    print('{} Epoch [{:03d}/{:03d}], MAE: {:.4f}, S-Measure: {:.4f}, Validation Loss: {:.4f}'.
          format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.epoch, mae.mean(), s.mean(), val_loss.mean()))

    return val_loss.mean()

def write_lr(optimizer, writer, step):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    writer.add_scalar('Learning Rate', float(lr), step)
    return lr

device = torch.device(args.device)
print('Device: {}'.format(device))

model = CPD.load_model(args.model).to(device)

optimizer = torch.optim.Adam(model.parameters(), args.lr)

transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gt_transform = transforms.Compose([
            transforms.Resize((args.imgres, args.imgres)),
            transforms.ToTensor()])


save_dir = os.path.join('training', model.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ckpt_path = os.path.join(save_dir, 'ckpts')
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

dataset = CPD.ImageGroundTruthFolder(args.datasets_path, transform=transform, target_transform=gt_transform, crop=args.rand_crop)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
val_dataset = CPD.ImageGroundTruthFolder('./datasets/val', transform=transform, target_transform=gt_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
train_writer = tensorboard.SummaryWriter(os.path.join(save_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'), 'train'))
val_writer = tensorboard.SummaryWriter(os.path.join(save_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'), 'val'))
print('Dataset loaded successfully')

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.lr_patience, threshold=0.01, verbose=True)
best_val_loss = validate(model, val_loader, val_writer, 1, 0)
patience = 0
for epoch in range(1, args.epoch+1):
    write_lr(optimizer, train_writer, (epoch-1)*(len(train_loader)+1))
    print('Started epoch {:03d}/{}'.format(epoch, args.epoch))
    train(model, train_loader, optimizer, epoch, train_writer)
    torch.save(model.state_dict(), '{}/{}.{:02d}.{:05d}.pth'.format(ckpt_path, model.name, epoch, epoch*len(train_loader)))
    val_loss = validate(model, val_loader, val_writer, epoch, epoch*len(train_loader))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
    else:
        patience += 1
        if patience >= args.training_patience:
            print('Training completed after {} epochs'.format(epoch))
            exit()
    lr_scheduler.step(val_loss)
