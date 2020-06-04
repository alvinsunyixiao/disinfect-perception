import numpy as np
import os
import torch
import torchvision

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from segmentation.augment import \
    MultiRandomAffineCrop, MultiCenterAffineCrop, ImageAugmentor
from utils.params import ParamDict as o

class SegEncoder:

    def __init__(self, num_classes):
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
            'loss_mask_b1hw': self.pil_to_tensor(tmp_dict['loss_mask']).float(),
        }

class COCODataset(Dataset):

    DEFAULT_PARAMS=o(
        version=2017,
        data_dir='/data/coco2017',
        annotation_dir='/data/coco2017/annotations',
        min_area=200,
        crop_params=MultiRandomAffineCrop.DEFAULT_PARAMS,
        augment_params=ImageAugmentor.DEFAULT_PARAMS,
        classes=set([
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
        self.p = params
        self.mode = 'train' if train else 'val'
        self.annotation_path = os.path.join(self.p.annotation_dir,
            'instances_{}{}.json'.format(self.mode, self.p.version))
        self.img_dir = os.path.join(self.p.data_dir,
            '{}{}'.format(self.mode, self.p.version))
        self.coco = COCO(self.annotation_path)
        self.img_ids = list(self.coco.imgs.keys())
        if train:
            self.multi_crop = MultiRandomAffineCrop(self.p.crop_params)
            self.img_augmentor = ImageAugmentor(self.p.augment_params)
        else:
            self.multi_crop = MultiCenterAffineCrop(self.p.crop_params)
            self.img_augmentor = torchvision.transforms.ToTensor()
        self.encoder = SegEncoder(len(self.p.classes))
        self.class_map = self._generate_class_map()

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
        loss_mask = torch.zeros_like(seg_mask)
        for ann in annotations:
            ann_mask = torch.from_numpy(self.coco.annToMask(ann))
            # mask indicating invalid regions
            if ann['iscrowd'] or ann['area'] < self.p.min_area:
                loss_mask = torch.bitwise_or(loss_mask, ann_mask)
            elif ann['category_id'] in self.class_map:
                class_id = self.class_map[ann['category_id']]
                seg_mask = torch.max(seg_mask, ann_mask*class_id)
        return {
            'image': self._get_img(img_id),
            'seg_mask': seg_mask,
            'loss_mask': loss_mask,
        }

    def __getitem__(self, key):
        raw_data = self.get_raw_data(key)
        crop_data = self.multi_crop(raw_data)
        crop_data['image'] = self.img_augmentor(crop_data['image'])
        enc_data = self.encoder(crop_data)
        return enc_data

    def __len__(self):
        return len(self.coco.imgs)

