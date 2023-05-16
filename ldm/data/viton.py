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

semantic_labels = [
    [85, 107, 47],
	[0, 0, 255],
	[255, 0,255],
	[219, 112, 147],
	[255, 20, 147],
	[0, 191, 255],
	[216, 191, 216],
	[30, 144, 255],
	[240, 230, 140],  
]
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

    
class VitonDataset(data.Dataset):
    
    def __init__(self, data_root,datamode="train",img_size=256,train_size=0.95):
        super(VitonDataset, self).__init__()
        
        self.db_path = data_root
        self.split = datamode
        self.train_size = train_size
        self.val_size = 1.0-train_size
        self.img_size = (int(img_size * 0.75),img_size)
        self.no_bg = False
        
        self.filepath_df = pd.read_csv(osp.join(self.db_path, "viton_%s_pairs.txt" % ("test" if self.split == "test" else "train")), sep=" ", names=["poseA", "target"])
        
        if datamode == "train":
            self.filepath_df = self.filepath_df.iloc[:int(len(self.filepath_df) * self.train_size)]
        elif datamode == "val":
            self.filepath_df = self.filepath_df.iloc[-int(len(self.filepath_df) * self.val_size):]
        
        pad_length = int(img_size * 0.25)//2
        self.transform = transforms.Compose([
            transforms.Pad(padding=(pad_length,0,pad_length,0)),
            # transforms.ToTensor()
        ])
        
    def name(self):
        return "VitonDataset"
    
    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]
        
        c_name = df_row["target"].split("/")[-1]
        im_name = df_row["poseA"].split("/")[-1]
        
        # get original image of person
        image = Image.open(osp.join(self.db_path, "data", "image", df_row["poseA"]))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize(self.img_size, resample=PIL.Image.LINEAR)
        # load cloth labels
        cloth_seg = Image.open(osp.join(self.db_path, "data", "image_parse_with_hands", df_row["poseA"].replace(".jpg", ".png")))
        if not cloth_seg.mode == "RGB":
            cloth_seg = cloth_seg.convert("RGB")
        cloth_seg = cloth_seg.resize(self.img_size, resample=PIL.Image.NEAREST)
        
        densepose = Image.open(osp.join(self.db_path, "data", "image_densepose_parse", df_row["poseA"].replace(".jpg", ".png")))
        if not densepose.mode == "RGB":
            densepose = densepose.convert("RGB")
        densepose = densepose.resize(self.img_size, resample=PIL.Image.NEAREST)
        
        # mask the image to get desired inputs
        
        # get the mask without upper clothes / dress, hands, neck
        # additionally, get cloth segmentations by cloth part
        cloth_seg = np.array(cloth_seg).astype(np.uint8)
        mask = np.zeros([self.img_size[1],self.img_size[0]])
        for i, color in enumerate(semantic_cloth_labels):
            if i < (6 + self.no_bg):    # this works, because colors are sorted in a specific way with background being the 8th.
                mask[np.all(cloth_seg == color, axis=-1)] = 1.0

        densepose = np.array(densepose).astype(np.uint8)
        for i, color in enumerate(semantic_labels):
            mask[np.all(densepose == color, axis=-1)] = 1.0
                
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1).astype(np.uint8)
        masked_image = image * (1 - mask)
        masked_image = Image.fromarray(masked_image)
        
        target_cloth_image = Image.open(osp.join(self.db_path, "data", "cloth", df_row["target"]))
        if not target_cloth_image.mode == "RGB":
            target_cloth_image = target_cloth_image.convert("RGB")
        target_cloth_image = target_cloth_image.resize(self.img_size, resample=PIL.Image.LINEAR)
        
        warped_cloth_image = Image.open(osp.join(self.db_path, "data", "warped_cloth", df_row["target"]))
        if not warped_cloth_image.mode == "RGB":
            warped_cloth_image = warped_cloth_image.convert("RGB")
        warped_cloth_image = warped_cloth_image.resize(self.img_size, resample=PIL.Image.LINEAR)
        
        
        # pad to 256*256
        image = self.transform(image)
        masked_image = self.transform(masked_image)
        target_cloth_image = self.transform(target_cloth_image)
        warped_cloth_image = self.transform(warped_cloth_image)
        
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)        
        masked_image = np.array(masked_image).astype(np.uint8)
        masked_image = (masked_image / 127.5 - 1.0).astype(np.float32)
        target_cloth_image = np.array(target_cloth_image).astype(np.uint8)
        target_cloth_image = (target_cloth_image / 127.5 - 1.0).astype(np.float32)
        warped_cloth_image = np.array(warped_cloth_image).astype(np.uint8)
        warped_cloth_image = (warped_cloth_image / 127.5 - 1.0).astype(np.float32)        
        
       
        result = {
            'c_name':               c_name,                     # for visualization
            'im_name':              im_name,                    # for visualization or ground truth
            'image':                image,                      # for visualization or ground truth
            
            'masked_image':         masked_image,               # for input
            'target_cloth':         target_cloth_image,         # for input
            'warped_cloth':         warped_cloth_image,         # for input
        }
        
        return result
    
    def __len__(self):
        return len(self.filepath_df)


class VitonDatasetTrain(VitonDataset):
    def __init__(self, **kwargs):
        super().__init__(data_root="datasets/viton",datamode="train", **kwargs)


class VitonDatasetValidation(VitonDataset):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="datasets/viton",datamode="val", **kwargs)
