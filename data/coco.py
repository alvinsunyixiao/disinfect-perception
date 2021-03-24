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

from data.data import BaseSet
import time
from data.label_unifier import get_coco_label_unifier

class COCODataset(BaseSet):
    DEFAULT_PARAMS = BaseSet.DEFAULT_PARAMS(
        version=2017,
        data_dir='/data/COCO2017',
        annotation_dir='/data/COCO2017/annotations',
        min_area=200,
        classes=set([
            'person',
            'bottle',
            'wine glass',
            'cup',
            'fork',
            'knife',
            'spoon',
            'bowl',
            'chair',
            'couch',
            'bed',
            'dining table',
            'toilet',
            'laptop',
            'mouse',
            'remote',
            'keyboard',
            'microwave',
            'oven',
            'toaster',
            'sink',
            'refrigerator',
        ]),
    )

    def __init__(self, params=DEFAULT_PARAMS, train=True):
        super(COCODataset, self).__init__(params, train)
        self.annotation_path = os.path.join(self.p.annotation_dir,
            'instances_{}{}.json'.format(self.mode, self.p.version))
        self.img_dir = os.path.join(self.p.data_dir,
            '{}{}'.format(self.mode, self.p.version))
        self.coco = COCO(self.annotation_path)
        self.img_ids = list(self.coco.imgs.keys())
        self.class_map = self._generate_class_map()
        self.label_dict = self.get_class_names()
        self.label_unifier, self.valid_label_idx = get_coco_label_unifier(self.label_dict)
    
    def get_class_names(self):
        idx = 1
        ret = {}
        for cat_id, cat in self.coco.cats.items():
            if cat['name'] in self.p.classes:
                # map cat['name'] to idx
                ret[idx] = cat['name']
                idx += 1
        return ret

    def _generate_class_map(self):
        idx = 1
        mapping = {}
        for cat_id, cat in self.coco.cats.items():
            if cat['name'] in self.p.classes:
                mapping[cat_id] = idx
                idx += 1
        return mapping

    def _get_img(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        img_fpath = os.path.join(self.img_dir, img_fname)
        return Image.open(img_fpath).convert('RGB')

    def get_raw_data(self, key):
        assert isinstance(key, int), "non integer key not supported!"
        img_id = self.img_ids[key]
        annotations = self.coco.imgToAnns[img_id]
        img = self._get_img(img_id)
        seg_mask = torch.zeros((img.size[1], img.size[0]), dtype=torch.uint8)
        loss_mask = torch.ones_like(seg_mask)
        for ann in annotations:
            ann_mask = torch.from_numpy(self.coco.annToMask(ann))
            # mask indicating invalid regions
            if ann['iscrowd'] or ann['area'] < self.p.min_area:
                loss_mask = torch.bitwise_and(loss_mask, torch.bitwise_not(ann_mask))
            elif ann['category_id'] in self.class_map:
                class_id = self.class_map[ann['category_id']]
                seg_mask = torch.max(seg_mask, ann_mask*class_id)
        seg_mask = self.label_unifier(seg_mask)
        return {
            'image': self._get_img(img_id),
            'seg_mask': seg_mask,
            'loss_mask': loss_mask,
            'valid_label_idx': self.valid_label_idx,
        }

    def __len__(self):
        return len(self.coco.imgs)