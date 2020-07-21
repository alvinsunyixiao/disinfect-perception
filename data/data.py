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

# Augmentation
from data.augment import \
    MultiRandomAffineCrop, MultiCenterAffineCrop, ImageAugmentor
# Subset classes from open source datasets
from data.label_params import fg_ade20k_coi
from data.label_unifier import \
    get_coco_label_unifier, get_fine_grained_ade_label_unifier, get_hospital_label_unifier
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
        color_jitter=o(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        )
    )

    def __init__(self, params, train):
        self.p = params
        self.mode = 'train' if train else 'val'
        if train:
            self.multi_crop = MultiRandomAffineCrop(self.p.crop_params)
            self.img_augmentor = ImageAugmentor(self.p.augment_params)
        else:
            self.multi_crop = MultiCenterAffineCrop(self.p.crop_params)
            self.img_augmentor = torchvision.transforms.ToTensor()
        self.encoder = SegEncoder(24) # TODO: use a consistent num_class variable
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=self.p.color_jitter.brightness,
            contrast=self.p.color_jitter.contrast,
            saturation=self.p.color_jitter.saturation,
            hue = self.p.color_jitter.hue
        )

    def get_raw_data(self, key):
        raise NotImplementedError("Pure Virtual Method")

    def __getitem__(self, key):
        raw_data = self.get_raw_data(key)
        crop_data = self.multi_crop(raw_data)
        crop_data['image'] = self.img_augmentor(crop_data['image'])
        enc_data = self.encoder(crop_data)
        enc_data['valid_label_idx'] = torch.tensor(enc_data['valid_label_idx'], dtype = torch.bool)
        return enc_data

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
        total_data_cnt = 167
    )

    def __init__(self, params=DEFAULT_PARAMS, train=True):
        '''
        Initialize and load the ADE20K annotation file into memory.
        '''
        super(HospitalDataset, self).__init__(params, train)
        img_path_template = os.path.join(self.p.root_dir, "{0}.jpg")
        json_path_template = os.path.join(self.p.root_dir, "{0}.json")
        self.img_path_list = []
        self.json_path_list = []
        for i in range(self.p.total_data_cnt):
            img_path = img_path_template.format(i)
            json_path = json_path_template.format(i)
            assert os.path.exists(img_path)
            assert os.path.exists(json_path)
            self.img_path_list.append(img_path)
            self.json_path_list.append(json_path)
        self.dataset_size = self.p.total_data_cnt # TODO: do some train-val split
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
        json_file = self.json_path_list[key]

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
            cur_len_sum += l
            if key < cur_len_sum:
                break # Select i-th dataset!
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