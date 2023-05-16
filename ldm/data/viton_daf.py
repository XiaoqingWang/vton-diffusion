#coding=utf-8
import json
import os.path as osp

import numpy as np
from numpy.lib.polynomial import polyfit
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import PIL
import json
import random
import imgaug.augmenters as iaa


# 'images' should be either a 4D numpy array of shape (N, height, width, channels)
# or a list of 3D numpy arrays, each having shape (height, width, channels).
# Grayscale images must have shape (height, width, 1) each.
# All images must have numpy's dtype uint8. Values are expected to be in
# range 0-255.
def vton_imgaug(image, img_agnostic, cloth, img_mask, img_mask_org, pose_mask):
    Temperature = np.random.randint(1000, 40000)
    brightness = np.random.rand() + 0.5
    Hue = np.random.rand() + 0.5
    R = np.random.rand() * 90 - 45
    colorseq = iaa.Sequential(
        [iaa.ChangeColorTemperature(Temperature),
         iaa.MultiplyBrightness(brightness),
         iaa.MultiplyHue(Hue)])
    flipseq = iaa.Fliplr()
    rotate = iaa.Rotate(R)

    image, img_agnostic, cloth = colorseq(images=[image, img_agnostic, cloth])
    

    if random.random() > 0.5:
        image, img_agnostic, cloth, img_mask, img_mask_org, pose_mask= flipseq(images=[image, img_agnostic, cloth,img_mask, img_mask_org, pose_mask])
    
    image, img_agnostic, cloth, img_mask, img_mask_org, pose_mask= rotate(images=[image, img_agnostic, cloth,img_mask, img_mask_org, pose_mask])
    
    return image, img_agnostic, cloth, img_mask, img_mask_org, pose_mask


class VitonDAFDataset(data.Dataset):

    def __init__(self, data_root, dataset_list='train.txt', datamode="train", img_size=256, test_muti_pos=False):
        super(VitonDAFDataset, self).__init__()
        self.img_size = (int(img_size * 0.75), img_size)

        self.data_path = data_root
        self.datamode = datamode
        self.dataset_list = dataset_list
        self.test_muti_pos = test_muti_pos
        pad_length = int(img_size * 0.25) // 2
        self.transform = transforms.Compose([
            transforms.Pad(padding=(pad_length, 0, pad_length, 0)),
            # transforms.ToTensor()
        ])
        img_names = []
        c_names = []
        pose_names = []

        with open(osp.join(self.data_path, self.dataset_list), 'r') as f:
            for line in f.readlines():
                if self.datamode == "train":
                    img_name = line.rstrip().replace("png", "jpg")
                    c_name = img_name.replace("_0.jpg", "_1.jpg")
                    pose_name = img_name.replace('.jpg', '_keypoints.jpg')
                elif self.test_muti_pos:
                    img_name, c_name, pose_name = line.strip().split()
                else:
                    img_name, c_name = line.strip().split()
                    pose_name = img_name.replace('.jpg', '_keypoints.jpg')
                img_names.append(img_name)
                c_names.append(c_name)
                pose_names.append(pose_name)

        self.img_names = img_names
        self.c_names = c_names
        self.pose_names = pose_names

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = self.c_names[index]
        pose_name = self.pose_names[index]
        c = Image.open(osp.join(self.data_path, 'clothes', c_name)).convert('RGB')
        pose_rgb = Image.open(osp.join(self.data_path, 'vis_pose', pose_name))
        img = Image.open(osp.join(self.data_path, self.datamode + '_img', img_name))
        img_agnostic = Image.open(osp.join(self.data_path, 'img_agnostic', img_name.replace("jpg", "png")))

        img_mask = img_agnostic.convert('L').point(lambda x: 0 if x >= 5 else 1, '1')
        pose_mask = pose_rgb.convert('L').point(lambda x: 0 if x < 15 else 1, '1')

        c = self.transform(c)
        pose_rgb = self.transform(pose_rgb)
        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)
        img_mask = self.transform(img_mask)
        pose_mask = self.transform(pose_mask)

        mask_size = (self.img_size[1] // 4, self.img_size[1] // 4)
        img_mask_org = np.array(img_mask)[..., None]
        img_mask = np.array(img_mask.resize(mask_size, resample=PIL.Image.NEAREST))[..., None]
        pose_mask = np.array(pose_mask.resize(mask_size, resample=PIL.Image.NEAREST))[..., None]

        if self.datamode == 'train' and random.random() > 0.3:
            img, img_agnostic, c, img_mask, img_mask_org, pose_mask = vton_imgaug(np.array(img), np.array(img_agnostic),
                                                                                  np.array(c), np.array(img_mask),
                                                                                  np.array(img_mask_org),
                                                                                  np.array(pose_mask))

        img = (np.array(img) / 127.5 - 1.0).astype(np.float32)
        img_agnostic = (np.array(img_agnostic) / 127.5 - 1.0).astype(np.float32)
        # pose_rgb = (np.array(pose_rgb) / 127.5 - 1.0).astype(np.float32)
        c = (np.array(c) / 127.5 - 1.0).astype(np.float32)
        img_mask = img_mask.astype(np.float32)
        pose_mask = pose_mask.astype(np.float32)
        img_mask_org = img_mask_org.astype(np.float32)

        result = {
            'img_name': img_name,
            'c_name': c_name,
            'pose_name': pose_name,
            'image': img.copy(),
            'img_agnostic': img_agnostic.copy(),
            # 'pose': pose_rgb.copy(),
            'cloth': c.copy(),
            'img_mask': img_mask.copy(),
            'img_mask_org': img_mask_org.copy(),
            'pose_mask': pose_mask.copy()
        }
        return result

    def __len__(self):
        return len(self.img_names)


class VitonDAFDatasetTrain(VitonDAFDataset):

    def __init__(self, **kwargs):
        super().__init__(data_root="datasets/VITON_DAF", datamode="train", dataset_list='train.txt', **kwargs)


class VitonDAFDatasetValidation(VitonDAFDataset):

    def __init__(self, **kwargs):
        super().__init__(data_root="datasets/VITON_DAF", datamode="test", dataset_list='test_pairs.txt', **kwargs)


class VitonDAFDatasetMultiPose(VitonDAFDataset):

    def __init__(self, **kwargs):
        super().__init__(data_root="datasets/VITON_DAF",
                         datamode="test",
                         dataset_list='test_multi_pos.txt',
                         test_muti_pos = True,
                         **kwargs)
