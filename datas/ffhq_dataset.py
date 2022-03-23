
from torch.utils.data import Dataset
import torchvision.transforms as transform
import torch
import numpy as np
import copy
import os
import random
from PIL import Image
import torchvision.datasets as dset
from random import randrange
import random
import sys
sys.path.append('..')
from utils.util import generate_stroke_mask
import glob


class FfhqDatasetMask(Dataset):
    
    def __init__(self, pt_dataset, image_size=32):
        self.dataset = []
        self.image_id_list = glob.glob(pt_dataset + '/images/*.png')
        for img_path in self.image_id_list:
            inpaint_mask_path = img_path.replace('/images/', '/inpainting_mask/')
            seg_mask_path = img_path.replace('/images/', '/segmentation_mask/')
            self.dataset.append((img_path, inpaint_mask_path, seg_mask_path))
        
        self.trans_resize = transform.Resize((image_size, image_size))

        self.trans_img = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])

        self.trans_mask = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean = (0.5), std = (0.5))
        ])

        print("## Image is %d"%(len(self.image_id_list)))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, inpaint_mask_path, seg_mask_path = self.dataset[idx]
        img = Image.open(img_path).convert('RGB')
        inpaint_mask = Image.open(inpaint_mask_path).convert('L')
        seg_mask = Image.open(seg_mask_path).convert('L')
        if self.trans_resize is not None:
            img = self.trans_resize(img)
            inpaint_mask = self.trans_resize(inpaint_mask)
            seg_mask = self.trans_resize(seg_mask)
        img_gt = copy.deepcopy(img)
        img = np.asarray(img)
        inpaint_mask = np.asarray(inpaint_mask)
        inpaint_mask[inpaint_mask < 127.5] = 0
        inpaint_mask[inpaint_mask >= 127.5] = 255
        img[inpaint_mask > 127.5] = 0
        img = Image.fromarray(img.astype(np.uint8))
        inpaint_mask = Image.fromarray(inpaint_mask.astype(np.uint8))
        img = self.trans_img(img)
        img_gt = self.trans_img(img_gt)
        inpaint_mask = self.trans_mask(inpaint_mask)
        seg_mask = self.trans_mask(seg_mask)
        return img, inpaint_mask, seg_mask, img_gt


class FfhqDatasetTest(Dataset):
    
    def __init__(self, pt_dataset, image_size=32):

        self.pt_dataset = pt_dataset
        self.image_id_list = glob.glob(pt_dataset + '/*/*.png')

        self.image_size=image_size
        self.trans_img = transform.Compose([
            transform.Resize((self.image_size, self.image_size)),
            transform.ToTensor(),
            transform.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])

        self.trans_mask = transform.Compose([
            transform.Resize((self.image_size, self.image_size)),
            transform.ToTensor(),
            transform.Normalize(mean = (0.5), std = (0.5))
        ])

        print("## Image is %d"%(len(self.image_id_list)))
        
    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        img_path = self.image_id_list[idx]
        inpaint_mask_path = img_path.replace('.png', '_mask.png')
        seg_mask_path = img_path.replace('.png', '_segm.png')
        img = Image.open(img_path).convert('RGB')
        inpaint_mask = Image.open(inpaint_mask_path).convert('L')
        seg_mask = Image.open(seg_mask_path).convert('L')
        inpaint_mask = np.asarray(inpaint_mask)
        inpaint_mask[inpaint_mask < 127.5] = 0
        inpaint_mask[inpaint_mask >= 127.5] = 255
        inpaint_mask = Image.fromarray(inpaint_mask.astype(np.uint8))
        img = self.trans_img(img)
        inpaint_mask = self.trans_mask(inpaint_mask)
        seg_mask = self.trans_mask(seg_mask)
        return img, inpaint_mask, seg_mask
