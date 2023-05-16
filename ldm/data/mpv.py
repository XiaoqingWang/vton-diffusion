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

semantic_cloth_labels = [
    [128, 0, 128],
    [128, 128, 64],
    [128, 128, 192],
    [0, 255, 0],
    [0, 128, 128], # dress
    [128, 128, 128], # something upper?
    
    [0, 0, 0], # bg
    
    [0, 128, 0], # hair
    [0, 64, 0], # left leg?
    [128, 128, 0], # right hand
    [0, 192, 0], # left foot
    [128, 0, 192], # head
    [0, 0, 192], # legs / skirt?
    [0, 64, 128], # skirt?
    [128, 0, 64], # left hand
    [0, 192, 128], # right foot
    [0, 0, 128],
    [0, 128, 64],
    [0, 0, 64],
    [0, 128, 192]
]

class MPVDataset(data.Dataset):
    
    def __init__(self, data_root,datamode="train",img_size=256,train_size=0.95, test_pairs=None):
        super(MPVDataset, self).__init__()
        
        self.db_path = data_root
        self.datamode = datamode
        self.train_size = train_size
        self.val_size = 1.0-train_size
        self.img_size = (int(img_size * 0.75),img_size)
        self.no_bg = False
        
        self.filepath_df = pd.read_csv(osp.join(self.db_path, "all_poseA_poseB_clothes.txt"), sep="\t", names=["poseA", "poseB", "target", "split"])
        self.filepath_df = self.filepath_df.drop_duplicates("poseA")
        self.filepath_df = self.filepath_df[self.filepath_df["poseA"].str.contains("front")]
        self.filepath_df = self.filepath_df.drop(["poseB"], axis=1)
        self.filepath_df = self.filepath_df.sort_values("poseA")

        if self.datamode in {"test", "test_same"}:
            self.filepath_df = self.filepath_df[self.filepath_df.split == "test"]
            
            if self.datamode == "test":
                
                if test_pairs is None:
                    del self.filepath_df["target"]
                    filepath_df_new = pd.read_csv(osp.join(self.db_path, "test_unpaired_images.txt"), sep=" ", names=["poseA", "target"])
                    self.filepath_df = pd.merge(self.filepath_df, filepath_df_new, how="left")
                else:
                    self.filepath_df = pd.read_csv(osp.join(self.db_path, test_pairs), sep=" ", names=["poseA", "target"])
            
        elif self.datamode in {"train", "val", "train_whole"}:
            self.filepath_df = self.filepath_df[self.filepath_df.split == "train"]
            
            if self.datamode == "train":
                self.filepath_df = self.filepath_df.iloc[:int(len(self.filepath_df) * self.train_size)]
            elif self.datamode == "val":
                self.filepath_df = self.filepath_df.iloc[-int(len(self.filepath_df) * self.val_size):]

        pad_length = int(img_size * 0.25)//2
        self.transform = transforms.Compose([
            transforms.Pad(padding=(pad_length,0,pad_length,0)),
            # transforms.ToTensor()
        ])
        
    def name(self):
        return "MPVDataset"
    
    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]
        
        c_name = df_row["target"].split("/")[-1]
        img_name = df_row["poseA"].split("/")[-1]
        pose_name = ''
        
        # get original image of person
        img = Image.open(osp.join(self.db_path, df_row["poseA"])).convert("RGB")
        pose_rgb = Image.open(osp.join(self.db_path, df_row["poseA"][:-4] + "_pose.png")).convert("RGB")
        img_agnostic = Image.open(osp.join(self.db_path, df_row["poseA"][:-4] + "_agnostic.png")).convert("RGB")
        c = Image.open(osp.join(self.db_path, df_row["target"])).convert("RGB")
        
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

        if self.datamode[:5] == 'train' and random.random() > 0.3:
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
        return len(self.filepath_df)


class MPVDatasetTrain(MPVDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="datasets/mpv",datamode="train_whole", **kwargs)


class MPVDatasetValidation(MPVDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="datasets/mpv",datamode="test", **kwargs)
