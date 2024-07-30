import cv2
from torch.utils.data import Dataset
#
import torch
import numpy as np
import cv2
import os
import pandas as pd
import pdb
from PIL import Image,ImageOps
# from torchvision.transforms import Resize
class data_moffitt(Dataset):

    def __init__(self,images_dir,anno_dir,transforms=None):

        self.images_dir = images_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        
        images_list = os.listdir(self.images_dir)
        images_list = sorted(images_list)
        annotations_list =  os.listdir(self.anno_dir)
        annotations_list = sorted(annotations_list)
        
        self.data_info  = pd.DataFrame({'images':images_list,'annotations':annotations_list})


    def __getitem__(self,index):
        # pdb.set_trace()
        patch_name = self.data_info.iloc[index,0]
        gt_name = self.data_info.iloc[index,1]
        patch = Image.open(os.path.join(self.images_dir,patch_name))
        gt = Image.open(os.path.join(self.anno_dir,gt_name))

        gt= ImageOps.grayscale(gt) 
        # pdb.set_trace()
        if self.transforms:
            # print(patch.shape)
            patch,gt = self.transforms(patch), self.transforms(gt)
            # patch = Resize(patch,256)
            # print(patch.shape)
        
        # gt = torch.tensor(gt,dtype=torch.float32)
        # pdb.set_trace()
        return patch,gt
    def __len__(self):
        
        return len(self.data_info)
    