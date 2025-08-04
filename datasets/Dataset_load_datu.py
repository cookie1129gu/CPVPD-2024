import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform

class RS_Process:
    def __init__(self, num_classes=2, colormap=None, mean=None, std=None):
        self.num_classes = num_classes
        self.COLORMAP = colormap or [[0, 0, 0], [255, 255, 255]]
        self.CLASSES = ['Background', 'Pv']
        self.MEAN = mean or np.array([108.46, 118.90, 117.04])
        self.STD = std or np.array([44.37, 45.13, 52.41])
        self.colormap2label = self._build_colormap2label()

    def _build_colormap2label(self):
        colormap2label = np.zeros(256 ** 3)
        for i, cm in enumerate(self.COLORMAP):
            colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return colormap2label

    def showIMG(self, img):
        plt.imshow(img)
        plt.show()

    def normalize_image(self, im):
        return (im - self.MEAN) / self.STD

    def normalize_images(self, imgs):
        for i, im in enumerate(imgs):
            imgs[i] = self.normalize_image(im)
        return imgs

    def Index2Color(self, pred):
        colormap = np.asarray(self.COLORMAP, dtype='uint8')
        x = np.asarray(pred, dtype='int32')
        return colormap[x, :]

    def Color2Index(self, ColorLabel):
        data = ColorLabel.astype(np.int32)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        IndexMap = self.colormap2label[idx]
        IndexMap = IndexMap * (IndexMap <= self.num_classes)
        return IndexMap.astype(np.uint8)

    def Colorls2Index(self, ColorLabels):
        for i, data in enumerate(ColorLabels):
            ColorLabels[i] = self.Color2Index(data)
        return ColorLabels

    def read_RSimages_nolabel(self, mode, root, rescale=False, rotate_aug=False):
        img_dir = root
        data_list = os.listdir(img_dir)

        data = []
        count = 0
        for it in data_list:
            it_name = it[:-4]
            if len(it_name) == 0:  # Skip empty filenames
                print("it" + it)
                continue
            it_ext = it[-4:]
            if it_name[0] == '.':
                continue
            if it_ext in ('.result', '.TIF'):
                img_path = os.path.join(img_dir, it)
                img = io.imread(img_path)
                data.append(img)
                count += 1
                if not count % 500:
                    print(f'{count}/{len(data_list)} images loaded.')
        print(f'{len(data)} {mode} images loaded.')
        return data

    def get_file_name(self, mode, root):
        mask_dir = root
        data_list = os.listdir(mask_dir)
        for vi, it in enumerate(data_list):
            data_list[vi] = it[:-4]
        return data_list


class RS(data.Dataset):
    def __init__(self, mode, root, random_flip=False, processor=None):
        self.mode = mode
        self.root = root
        self.random_flip = random_flip
        self.processor = processor or RS_Process()
        self.data = self.processor.read_RSimages_nolabel(mode, root, rescale=False)
        self.len = len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.random_flip:
            data = transform.rand_flip(data)
        data = self.processor.normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data

    def __len__(self):
        return self.len
