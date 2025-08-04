import math
import os
import time
import torch
import torch.autograd
from matplotlib import pyplot as plt
from skimage import io
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

import Eval.pre.train.datasets.Dataset_load as RS
from Eval.pre.model.DSFA_SwinNet.DSFA_SwinNet import DSFA_SwinNet as Net
from Eval.pre.model.Loss.CombinedLoss import CombinedLoss as main_loss
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter

# Get computing device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

NET_NAME = 'DSFA_SwinNet'
DATA_NAME = 'google(DSFA_SwinNet)'
working_path = os.path.abspath('.')

args = {
    'train_batch_size': 16,
    'val_batch_size': 16,
    'train_crop_size': 256,
    'val_crop_size': 256,
    'epochs': 100,
    'gpu': device != 'cpu',
    'lr': 0.0602,
    'weight_decay': 4e-4,
    'momentum': 0.6665,
    'print_freq': 100,
    'predict_step': 5,
    'weight_lovasz': 0,
    'weight_bce': 1,
    'weight_dice': 0.9838,
    'flooding': 1,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'XX.pth')
}

loss_weights = {
    'w16': 1,
    'w32': 1,
    'w64': 1,
    'w128': 1,
    'w256': 1,
}

# Create directories if they don't exist
for dir_path in [args['log_dir'], args['chkpt_dir'], args['pred_dir']]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

writer = SummaryWriter(args['log_dir'])


def main():
    print(RS.num_classes)

    # Initialize network with 3 input channels and 2 output classes
    net = Net(num_classes=RS.num_classes)
    net = net.to(device)

    # Prepare datasets and dataloaders
    train_set = RS.RS('train', random_flip=True)
    val_set = RS.RS('val')

    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    # Loss function and optimizer
    main_criterion = main_loss()
    main_criterion = main_criterion.to(device)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=0.1,
        weight_decay=args['weight_decay'],
        momentum=args['momentum'],
        nesterov=True
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

    # Start training
    train(train_loader, net, main_criterion, optimizer, scheduler, 0, args, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, main_criterion, optimizer, scheduler, curr_epoch, train_args, val_loader):
    # Visualization metrics
    Acc = []
    train_F1 = []
    val_F1 = []
    train_loss = []
    val_loss = []
    cu_ep = []

    bestaccT = 0
    bestF = 0
    bestIoU = 0
    bestloss = 1
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])

    while True:
        # Clear GPU cache
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        F1_meter = AverageMeter()
        train_main_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)

        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_learning_rate(optimizer, running_iter, all_iters, args)
            imgs, labels = data

            if args['gpu']:
                imgs = imgs.cuda().float()
                labels = labels.cuda().long()
            else:
                imgs = imgs.float()
                labels = labels.long()

            # Check for NaN values in data
            if torch.isnan(imgs).any():
                print("Warning: NaN values found in data at iteration:", running_iter)
            if torch.isnan(labels).any():
                print("Warning: NaN values found in labels at iteration:", running_iter)

            # Forward pass
            optimizer.zero_grad()
            features = net(imgs)
            outputs = features[-1]

            assert outputs.shape[1] == RS.num_classes

            # Check for NaN in model outputs
            if torch.isnan(outputs).any():
                print("Warning: NaN values found in model outputs at iteration:", running_iter)

            # Calculate loss with flooding
            loss = main_criterion(outputs, labels)
            loss = (loss - args['flooding']).abs() + args['flooding']
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()

            _, preds = torch.max(outputs, dim=1)
            preds = preds.numpy()

            F1_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, IoU = accuracy(pred, label)
                if F1 > 0:
                    F1_curr_meter.update(F1)

            F1_meter.update(F1_curr_meter.avg if F1_curr_meter.avg is not None else 0)
            train_main_loss.update(loss.cpu().detach().numpy())

            # Print training status
            if (i + 1) % train_args['print_freq'] == 0:
                curr_time = time.time() - start
                print(f'[epoch {curr_epoch}] [iter {i + 1} / {len(train_loader)} {curr_time:.1f}s] '
                      f'[lr {optimizer.param_groups[0]["lr"]:.6f}] '
                      f'[train loss {train_main_loss.val:.4f} F1 {F1_meter.avg * 100:.2f}]')

                writer.add_scalar('train loss', train_main_loss.val, running_iter)
                writer.add_scalar('train F1', F1_meter.avg, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        # Validation
        val_F, val_acc, val_IoU, loss_v = validate(val_loader, net, main_criterion)

        # Update metrics
        Acc.append(val_acc)
        val_F1.append(val_F)
        val_loss.append(loss_v)
        train_loss.append(train_main_loss.avg)
        train_F1.append(F1_meter.avg)
        cu_ep.append(curr_epoch)

        # Save best model
        if val_F > bestF:
            bestF = val_F
            bestloss = loss_v
            bestIoU = val_IoU
            save_path = os.path.join(args['chkpt_dir'],
                                     f'{NET_NAME}_{curr_epoch}e_OA{val_acc * 100:.2f}_F{val_F * 100:.2f}_IoU{val_IoU * 100:.2f}.pth')
            torch.save(net.state_dict(), save_path)

        print(f'Total time: {time.time() - begin_time:.1f}s Best rec: Val {bestF * 100:.2f}, '
              f'Val_loss {bestloss:.4f} BestIOU: {bestIoU * 100:.2f}')

        curr_epoch += 1

        # Check if training completed
        if curr_epoch >= train_args['epochs']:
            print(f'Acc {Acc}')
            print(f'train_F1 {train_F1}')
            print(f'val_F1 {val_F1}')
            print(f'train_loss {train_loss}')
            print(f'val_loss {val_loss}')

            # Save experimental records
            with open('experimental_record/LOSS_weight1/loss_main 0.2_0.8loss_aux.txt', 'a') as f:
                f.write(
                    f'Acc:{Acc}\n train_F1:{train_F1}\n val_F1:{val_F1}\n train_loss:{train_loss}\n val_loss:{val_loss}\n')

            # Plot metrics
            x_all = cu_ep
            plt.plot(x_all, val_loss, color='green', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='val_loss')
            plt.plot(x_all, train_loss, color='red', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='train_loss')
            plt.plot(x_all, Acc, color='blue', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='Acc')
            plt.plot(x_all, val_F1, color='black', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='val_F1')
            plt.plot(x_all, train_F1, color='yellow', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='train_F1')

            plt.legend()
            plt.savefig("loss_main 0.2_0.8loss_aux.png")
            plt.clf()
            return


def validate(val_loader, model_seg, main_criterion, save_pred=True):
    model_seg.eval()
    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    start = time.time()

    for vi, data in enumerate(val_loader):
        imgs, labels = data

        if args['gpu']:
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()
        else:
            imgs = imgs.float()
            labels = labels.long()

        with torch.no_grad():
            features = model_seg(imgs)
            out = features[-1]

            loss = 0
            for feature in features:
                loss += main_criterion(feature, labels)

            loss = (loss - args['flooding']).abs() + args['flooding']

        val_loss.update(loss.cpu().detach().numpy())

        # Calculate metrics
        out = out.cpu().detach()
        labels = labels.cpu().detach().numpy()

        _, preds = torch.max(out, dim=1)
        preds = preds.numpy()

        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)

        # Save prediction sample
        if save_pred and vi == 0:
            pred_color = RS.Index2Color(preds[0].squeeze())
            io.imsave(os.path.join(args['pred_dir'], f'{NET_NAME}.png'), pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print(
        f'{curr_time:.1f}s Val loss: {val_loss.average():.2f}, F1: {F1_meter.avg * 100:.2f}, Accuracy: {Acc_meter.average() * 100:.2f}')

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg


def adjust_learning_rate(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 1.5)
    running_lr = args['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    start = time.time()
    main()
