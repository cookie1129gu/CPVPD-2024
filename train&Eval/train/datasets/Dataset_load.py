import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform

# RS stands for Remote Sensing
num_classes = 2
COLORMAP = [[0, 0, 0], [255, 255, 255]]
CLASSES = ['Background', 'Pv']

# Mean and standard deviation for PV dataset (CPVPD)
MEAN = np.array([147.19, 148.08, 149.36])
STD = np.array([79.53, 79.99, 79.02])
root = 'D:/desk/PV/ImageProcess/Eval/pre/train/data/CPVPD_v2'


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im):
    return (im - MEAN) / STD


def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


# Create a color to index lookup table
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def Colorls2Index(ColorLabels):
    for i, data in enumerate(ColorLabels):
        ColorLabels[i] = Color2Index(data)
    return ColorLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)


def rescale_images(imgs, scale, order):
    for i, im in enumerate(imgs):
        imgs[i] = rescale_image(im, scale, order)
    return imgs


def rescale_image(img, scale=1 / 8, order=0):
    flag = cv2.INTER_NEAREST
    if order == 1:
        flag = cv2.INTER_LINEAR
    elif order == 2:
        flag = cv2.INTER_AREA
    elif order > 2:
        flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)),
                             interpolation=flag)
    return im_rescaled


def get_file_name(mode):
    data_dir = root
    mask_dir = os.path.join(data_dir, mode, 'images')

    data_list = os.listdir(mask_dir)
    for vi, it in enumerate(data_list):
        data_list[vi] = it[:-4]
    return data_list


def read_RSimages(mode, rescale=False, rotate_aug=False):
    data_dir = root
    img_dir = os.path.normpath(os.path.join(data_dir, mode, 'images'))
    mask_dir = os.path.normpath(os.path.join(data_dir, mode, 'label'))

    data_list = os.listdir(img_dir)

    data, labels = [], []
    count = 0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if it_name[0] == '.':
            continue
        if it_ext == '.bmp' or it_ext == '.png':
            img_path = os.path.join(img_dir, it)
            mask_path = os.path.join(mask_dir, it_name + it_ext)

            img = io.imread(img_path)
            label = io.imread(mask_path)

            if np.isnan(np.sum(img)) or np.isnan(np.sum(label)):
                print("Warning: NaN values found in image or label:", it)
                print("path : ", img_path)
                continue

            data.append(img)
            labels.append(label)

            count += 1
            if not count % 500:
                print('%d/%d images loaded.' % (count, len(data_list)))

    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')

    return data, labels


def read_RSimages_nolabel(mode, rescale=False, rotate_aug=False):
    data_dir = root
    img_dir = os.path.join(data_dir, mode, 'images')

    data_list = os.listdir(img_dir)

    data = []
    count = 0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if it_name[0] == '.':
            continue
        if it_ext == '.bmp':
            img_path = os.path.join(img_dir, it)
            img = io.imread(img_path)
            data.append(img)
            count += 1
            if not count % 500:
                print('%d/%d images loaded.' % (count, len(data_list)))

    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    return data


class RS(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.mode = mode
        self.random_flip = random_flip
        data, labels = read_RSimages(mode, rescale=False)

        self.data = data
        self.labels = Colorls2Index(labels)
        self.len = len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        if self.random_flip:
            data, label = transform.rand_flip(data, label)

        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))

        return data, label

    def __len__(self):
        return self.len
