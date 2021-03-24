import numpy as np
import os
import torch
import torchvision
import json

from copy import deepcopy
from labelme import utils
from scipy.io import loadmat
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import time

from data.data import BaseSet
from data.label_params import fg_ade20k_coi
from data.label_unifier import get_fine_grained_ade_label_unifier

class FineGrainedADE20KDataset(BaseSet):
    '''
    Fine-grained instance-level segmentation data from the 2016 ADE20K challenge.

    Data can be grabbed from https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
    '''
    DEFAULT_PARAMS = BaseSet.DEFAULT_PARAMS(
        root_dir = "/data/ADE20K_2016_07_26/",
        classes = fg_ade20k_coi
    )

    def __init__(self, params=DEFAULT_PARAMS, train=True):
        '''
        Initialize and load the ADE20K annotation file into memory.
        '''
        super(FineGrainedADE20KDataset, self).__init__(params, train)
        self.ds = loadmat(os.path.join(self.p.root_dir, "index_ade20k.mat"))
        self.ds = self.ds['index']
        self.data_dir_base = os.path.join(self.p.root_dir, '..')
        img_set_mark = 'train' if train else 'val'
        self.img_path_list = []
        self.seg_path_list = []
        for i in range(self.ds['filename'][0, 0].shape[1]):
            cur_file_name = self.ds['filename'][0, 0][0, i][0]
            if img_set_mark in cur_file_name:
                folder_path = self.ds['folder'][0, 0][0, i][0]
                img_path = os.path.join(self.data_dir_base, folder_path, cur_file_name)
                seg_path = FineGrainedADE20KDataset.get_seg_path(img_path)
                self.img_path_list.append(img_path)
                self.seg_path_list.append(seg_path)
        self.dataset_size = len(self.img_path_list)
        self.class_map = self._generate_class_map()
        self.label_dict = self.get_class_names()
        self.label_unifier, self.valid_label_idx = get_fine_grained_ade_label_unifier(self.label_dict)
    
    def get_class_names(self):
        class_name_file = os.path.join(self.p.root_dir, 'clustered_labels.txt')
        with open(class_name_file) as f:
            class_name_list = f.readlines()
        processed_label_list = [c.strip('\n') for c in class_name_list]
        ret = {}
        for i in range(len(processed_label_list)):
            ret[i + 1] = processed_label_list[i]
        return ret

    def get_raw_data(self, key, save_processed_image = False):
        """
        Args:
            key (int): key

        Returns:
            ret_dict
        """
        assert isinstance(key, int), "non integer key not supported!"
        img_path = self.img_path_list[key]
        seg_path = self.seg_path_list[key]
        processed_seg_image_path = seg_path[:-4] + '_processed.png'
        raw_img = Image.open(img_path).convert('RGB')
        if os.path.exists(processed_seg_image_path):
            # Read from pre-processed images
            seg_mask = np.array(Image.open(processed_seg_image_path), dtype = np.uint8)
        else:
            # Start from scratch
            seg_img = np.array(Image.open(seg_path), dtype = np.uint8)
            cat_map = seg_img[:,:,0] // 10
            cat_map = cat_map.astype(np.int)
            cat_map = cat_map * 256
            cat_map = cat_map + seg_img[:,:,1]
            seg_mask = self.class_map(cat_map).astype(np.uint8)
            if save_processed_image:
                processed_seg_image = Image.fromarray(seg_mask)
                processed_seg_image.save(new_seg_path)
        seg_mask = torch.tensor(seg_mask, dtype = torch.uint8)
        seg_mask = self.label_unifier(seg_mask)
        loss_mask = torch.ones_like(seg_mask)
        return {
            'image': raw_img,
            'seg_mask': seg_mask,
            'loss_mask': loss_mask,
            'valid_label_idx': self.valid_label_idx,
        }

    def _generate_class_map(self):
        class_name_list = []
        for i in range(self.ds['objectnames'][0, 0].shape[1]):
            class_name = self.ds['objectnames'][0, 0][0, i][0]
            class_name_list.append(class_name)
        # Take subset of class
        map_dict = {}
        cur_idx = 1 # Background maps to 0
        # Class of Interest
        for coi in self.p.classes:
            if not isinstance(coi, tuple):
                coi = (coi,)
            for c in coi:
                catmap_cls_id = class_name_list.index(c) + 1
                map_dict[catmap_cls_id] = cur_idx
            cur_idx += 1
        # Factory map function
        def map_func(elem):
            if elem in map_dict:
                return map_dict[elem]
            else:
                return 0 # Map everything else to zero
        vectorized_map_func = np.vectorize(map_func)
        return vectorized_map_func

    def __len__(self):
        return self.dataset_size
    
    @staticmethod
    def get_seg_path(img_path):
        return img_path[:-4] + '_seg.png'