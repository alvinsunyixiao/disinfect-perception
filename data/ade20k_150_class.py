import numpy as np
import os
import torch
import torchvision
import json
import csv

from copy import deepcopy
from labelme import utils
from scipy.io import loadmat
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import time

from data.data import BaseSet
from data.label_unifier import get_coarse_grained_ade_label_unifier

class CoarseGrainedADE20KDataset(BaseSet):
    '''
    Coarse grain semantic segmentation dataset from the 2016 ADE20K challenge.

    Data can be grabbed from http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    '''
    DEFAULT_PARAMS = BaseSet.DEFAULT_PARAMS(
        root_dir = "/data/ade20k_coarse/",
        train_json = "training.odgt",
        val_json = "validation.odgt",
        class_name_file = "object150_info.csv",
        skip_validation = True
    )

    def __init__(self, params=DEFAULT_PARAMS, train=True):
        '''
        Initialize and load the ADE20K annotation file into memory.
        '''
        super(CoarseGrainedADE20KDataset, self).__init__(params, train)

        if train:
            dataset_json_path = os.path.join(self.p.root_dir, self.p.train_json)
        else:
            dataset_json_path = os.path.join(self.p.root_dir, self.p.val_json)

        self.ds = [json.loads(x.rstrip()) for x in open(dataset_json_path, 'r')]

        self.dataset_size = len(self.ds)

        # For disinfection project, we do not want to do validation on the ADE20K
        # dataset (since all we care about is performance on hospital images)
        if not train and self.p.skip_validation:
            self.dataset_size = 0

        # Process label name and unify labels across different datasets.
        self.label_dict = self.get_class_names()
        self.label_unifier, self.valid_label_idx = get_coarse_grained_ade_label_unifier(self.label_dict)
    
    def get_class_names(self):
        class_name_file = os.path.join(self.p.root_dir, self.p.class_name_file)
        class_name_list = []
        with open(class_name_file, newline = '') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_name_list.append(row['Name'])
        ret = {}
        for i in range(len(class_name_list)):
            ret[i + 1] = class_name_list[i]
        return ret

    def get_raw_data(self, key, save_processed_image = False):
        """
        Args:
            key (int): key

        Returns:
            ret_dict
        """
        assert isinstance(key, int), "non integer key not supported!"
        img_path = os.path.join(self.p.root_dir, self.ds[key]['fpath_img'])
        seg_path = os.path.join(self.p.root_dir, self.ds[key]['fpath_segm'])
        raw_img = Image.open(img_path).convert('RGB')
        segm = Image.open(seg_path)
        assert(segm.mode == "L")
        assert(raw_img.size[0] == segm.size[0])
        assert(raw_img.size[1] == segm.size[1])
        seg_mask = torch.tensor(np.array(segm, dtype = np.uint8), dtype = torch.uint8)
        seg_mask = self.label_unifier(seg_mask)
        # seg_mask = self.label_unifier(seg_mask)
        loss_mask = torch.ones_like(seg_mask)
        return {
            'image': raw_img,
            'seg_mask': seg_mask,
            'loss_mask': loss_mask,
            'valid_label_idx': self.valid_label_idx,
        }

    def __len__(self):
        return self.dataset_size