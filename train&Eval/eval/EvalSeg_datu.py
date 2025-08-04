import math
import os
import time
import torch
import numpy as np
import torch.autograd
import torchvision
from matplotlib import pyplot as plt
from skimage import io
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from model.DSFA_SwinNet.DSFA_SwinNet import DSFA_SwinNet as Net

####################################

# Get computing hardware
# Use GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

#################################
DATA_NAME = 'CPVPD_v2(DSFA_SwinNet)'
#################################
# CPVPD_v2
load_path = 'D:/desk/PV/ImageProcess/Eval/pre/train/checkpoints/CPVPD_v2(DSFA_SwinNet)/DSFA_SwinNet_59e_OA92.75_F89.13_IoU80.99.pth'

working_path = os.path.dirname(os.path.abspath(__file__))
args = {
    'gpu': device != 'cpu',
    'batch_size': 1,
    'net_name': 'DSFA_SwinNet',
    'load_path': load_path
}

def soft_argmax(seg_map):
    assert seg_map.dim() == 4

    alpha = 1000.0
    b, c, h, w, = seg_map.shape
    print(seg_map.shape)
    soft_max = F.softmax(seg_map * alpha, dim=1)
    return soft_max


def evalSeg_datu(RS, pred_path):
    net = Net(num_classes=RS.processor.num_classes)
    print("RS.processor.num_classes:")
    print(RS.processor.num_classes)
    net = net.to(device)

    net.load_state_dict(torch.load(args['load_path']), strict=False)

    net.eval()  # Sets the model to evaluation mode
    print('Model loaded.')

    # pred_path = os.path.join(RS.root, 'pred', args['net_name'])
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    pred_name_list = RS.processor.get_file_name('test', RS.root)
    test_set = RS
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], num_workers=4, shuffle=False)
    predict(RS, net, test_loader, pred_path, pred_name_list)


def predict(RS, net, pred_loader, pred_path, pred_name_list):
    """total_iter is the total number of iterations for the data, i.e., the number of batches in the data loader"""
    total_iter = len(pred_loader)
    """num_files is the number of files in pred_name_list"""
    num_files = len(pred_name_list)
    # crop_nums = int(total_iter/num_files)  original

    crop_nums = math.ceil(total_iter / num_files)

    for vi, data in enumerate(pred_loader):
        imgs = data
        if args['gpu']:
            imgs = imgs.cuda().float()
        else:  # CPU training
            imgs = imgs.float()

        with torch.no_grad():
            features = net(imgs)
            outputs = features[-1]
            print(outputs.shape)
        output = outputs.detach().cpu()
        output = torch.argmax(output, dim=1)
        outputs = output.numpy()

        for i in range(args['batch_size']):
            idx = vi * args['batch_size'] + i
            file_idx = int(idx / crop_nums)
            crop_idx = idx % crop_nums
            if (idx >= total_iter):
                break
            pred = outputs[i]

            print(pred.shape)
            pred_color = RS.processor.Index2Color(pred.squeeze())
            if crop_nums > 1:
                pred_name = os.path.join(pred_path, pred_name_list[file_idx] + '_%d.TIF' % crop_idx)
            else:
                pred_name = os.path.join(pred_path, pred_name_list[file_idx] + '.TIF')
            io.imsave(pred_name, pred_color)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
