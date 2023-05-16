#coding=utf-8
from ldm.data.viton_daf import VitonDAFDatasetTrain
from ldm.data.mpv import MPVDatasetTrain
import torch.utils.data as data
from torch.utils.data.dataset import ConcatDataset

class ConcatMpvVitonTrain(data.Dataset):
    def __init__(self):
        super(ConcatMpvVitonTrain, self).__init__()
        a = VitonDAFDatasetTrain()
        b = MPVDatasetTrain()
        self.c = ConcatDataset([a,b])
        
    def __getitem__(self, index):
        return self.c.__getitem__(index)
        
    def __len__(self):
        return self.c.__len__()

