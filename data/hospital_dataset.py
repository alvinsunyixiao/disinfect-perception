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
from data.label_unifier import get_hospital_label_unifier

class HospitalDataset(BaseSet):
    '''
    Custom Hospital Dataset
    '''
    DEFAULT_PARAMS = BaseSet.DEFAULT_PARAMS(
        root_dir = "/data/hospital_images/",
        classes=[
            "hospital_bed",
            "seat",
            "medical_device",
            "screen",
            "bedrail",
            "floor",
            "wall",
            "ceil",
            "table",
            "door",
            "trolley",
            "outlet",
            "people",
            "pole",
            "cabinet",
            "sink",
            "window",
            "curtain",
            "faucet"
        ],
        total_data_cnt = 167,
        train_split = 110,
        val_split = 57,
    )

    def __init__(self, params=DEFAULT_PARAMS, train=True):
        '''
        Initialize and load the ADE20K annotation file into memory.
        '''
        super(HospitalDataset, self).__init__(params, train)
        json_path_template = os.path.join(self.p.root_dir, "{0}.json")
        self.json_path_list = []
        for i in range(self.p.total_data_cnt):
            json_path = json_path_template.format(i)
            assert os.path.exists(json_path)
            self.json_path_list.append(json_path)
        self.train = train
        if train:
            self.dataset_size = self.p.train_split
        else:
            self.dataset_size = self.p.val_split
        self.label_dict = self.get_class_names()
        self.label_name_to_value = {'_background_': 0}
        for num in self.label_dict:
            cls_name = self.label_dict[num]
            self.label_name_to_value[cls_name] = num
        self.label_unifier, self.valid_label_idx = get_hospital_label_unifier(self.label_dict)
    
    def get_class_names(self):
        processed_label_list = self.p.classes
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
        assert isinstance(key, int)
        if self.train:
            assert key >= 0 and key < self.p.train_split, "Got invalid key: {}".format(key)
            json_file = self.json_path_list[key]
        else:
            assert key >= 0 and key < self.p.val_split
            json_idx = -self.p.val_split + key
            json_file = self.json_path_list[json_idx]

        data = json.load(open(json_file))
        imageData = data["imageData"]
        raw_img = utils.img_b64_to_arr(imageData) # img: H x W x C
        seg_mask, _ = utils.shapes_to_label(raw_img.shape, data["shapes"], self.label_name_to_value) # HxW
        seg_mask = torch.tensor(seg_mask, dtype = torch.uint8)
        seg_mask = self.label_unifier(seg_mask)
        loss_mask = torch.ones_like(seg_mask)
        raw_img = Image.fromarray(raw_img)
        return {
            'image': raw_img,
            'seg_mask': seg_mask,
            'loss_mask': loss_mask,
            'valid_label_idx': self.valid_label_idx,
        }

    def __len__(self):
        return self.dataset_size