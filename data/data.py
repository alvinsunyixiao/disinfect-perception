import numpy as np
import os
import torch
import torchvision
import json
import random
from copy import deepcopy
from labelme import utils
from scipy.io import loadmat
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import time

# Augmentation
from data.augment import \
    MultiRandomAffineCrop, MultiCenterAffineCrop, ImageAugmentor
# Subset classes from open source datasets
from utils.params import ParamDict as o

class SegEncoder:

    def __init__(self, num_classes):
        """__init__.

        Args:
            num_classes:    number of classes
        """
        self.num_classes = num_classes

    def catgory_to_onehot(self, cat_map_pil):
        cat_map_1hw = self.pil_to_tensor(cat_map_pil)
        cat_ids_n = torch.arange(1, self.num_classes+1, dtype=cat_map_1hw.dtype)
        cat_ids_n11 = cat_ids_n[:, None, None]
        return (cat_ids_n11 == cat_map_1hw).float()

    def pil_to_tensor(self, pil_img):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pil_img.tobytes()))
        img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
        img = img.permute((2, 0, 1)).contiguous()
        return img

    def __call__(self, data_dict):
        tmp_dict = data_dict.copy()

        return {
            'image_b3hw': tmp_dict['image'],
            'seg_mask_bnhw': self.catgory_to_onehot(tmp_dict['seg_mask']),
            'loss_mask_bnhw': self.pil_to_tensor(tmp_dict['loss_mask']).float(),
            'valid_label_idx': tmp_dict['valid_label_idx'],
        }

class BaseSet(Dataset):
    '''
    Abstract base set

    To use this dataset implementation, simply inherent this class and
    implement the following methods:
        - __init__: overwrite init to implement necessary initialization.
            But don't forget to invoke parent constructor as well!
        - get_raw_data(self, key): return a dictionary comprising of data
            at dataset[key].
        - __len__: return the size of the underlying dataset.
    '''
    DEFAULT_PARAMS = o(
        crop_params=MultiRandomAffineCrop.DEFAULT_PARAMS,
        augment_params=ImageAugmentor.DEFAULT_PARAMS,
        horizontal_flip_prob = 0.5,
        num_classes = 24,
    )

    def __init__(self, params, train):
        self.p = params
        self.mode = 'train' if train else 'val'
        self.flip_prob = self.p.horizontal_flip_prob
        if train:
            self.multi_crop = MultiRandomAffineCrop(self.p.crop_params)
            self.img_augmentor = ImageAugmentor(self.p.augment_params)
        else:
            self.multi_crop = MultiCenterAffineCrop(self.p.crop_params)
            self.img_augmentor = torchvision.transforms.ToTensor()
        self.encoder = SegEncoder(self.p.num_classes)

    def get_raw_data(self, key):
        raise NotImplementedError("Pure Virtual Method")

    def __getitem__(self, key):
        raw_data = self.get_raw_data(key)
        crop_data = self.multi_crop(raw_data)
        if random.random() < self.flip_prob:
            # flip raw image/loss mask/label mask
            crop_data['image'] = crop_data['image'].transpose(Image.FLIP_LEFT_RIGHT)
            crop_data['seg_mask'] = crop_data['seg_mask'].transpose(Image.FLIP_LEFT_RIGHT)
            crop_data['loss_mask'] = crop_data['loss_mask'].transpose(Image.FLIP_LEFT_RIGHT)
        crop_data['image'] = self.img_augmentor(crop_data['image'])
        enc_data = self.encoder(crop_data)
        enc_data['valid_label_idx'] = torch.tensor(enc_data['valid_label_idx'], dtype = torch.bool)
        return enc_data

class DatasetMixer(object):
    def __init__(self, params, train = True):
        assert len(params) >= 1
        self.all_datasets = []
        for _cls, _param in params:
        #     print(_cls)
        #     assert isinstance(_cls, BaseSet)
        #     assert isinstance(_param, o)
            cls_instance = _cls(_param, train)
            self.all_datasets.append(cls_instance)
        self.dataset_len_list = []
        for ds in self.all_datasets:
            ds_len = len(ds)
            self.dataset_len_list.append(ds_len)
        self.total_ds_len = np.sum(self.dataset_len_list)

    def __getitem__(self, key):
        assert isinstance(key, int)
        cur_len_sum = 0
        for i in range(len(self.dataset_len_list)):
            l = self.dataset_len_list[i]
            if key < cur_len_sum + l:
                break # Select i-th dataset!
            cur_len_sum += l
        real_key = key - cur_len_sum
        return self.all_datasets[i][real_key]

    def __len__(self):
        return self.total_ds_len

if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    ds = FineGrainedADE20KDataset(train = False)
    for i in tqdm(range(len(ds))):
        output_dict = ds.get_raw_data(i)

# Load datasets
from data.fine_grained_ade20k import FineGrainedADE20KDataset
from data.hospital_dataset import HospitalDataset
from data.coco import COCODataset
from data.ade20k_150_class import CoarseGrainedADE20KDataset