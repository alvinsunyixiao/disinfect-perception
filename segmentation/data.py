import numpy as np
import os
import torch
import torchvision

from scipy.io import loadmat
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from utils.params import ParamDict as o
from segmentation.augment import MultiRandomAffineCrop, MultiCenterAffineCrop

def get_seg_filename(img_name):
    return img_name[:-4] + "_seg.png"

class SegEncoder:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def catgory_to_onehot(self, cat_map_1hw):
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
        # convert all PIL image to torch Tensors
        for key in tmp_dict:
            if isinstance(tmp_dict[key], Image.Image):
                tmp_dict[key] = self.pil_to_tensor(tmp_dict[key])

        return {
            'image_b3hw': tmp_dict['image'].float().div(255),
            'seg_mask_bnhw': self.catgory_to_onehot(tmp_dict['seg_mask']),
            'loss_mask_b1hw': tmp_dict['loss_mask'].float(),
        }

class BaseSet(Dataset):
    '''
    Abstract base set

    To use this dataset implementation, simply inherent this class and
    implement the following methods:
        - __init__: overwrite init to implement necessary initialization.
            But don't forget to invoke parent constructor as well!
        - get_raw_data(self, ind): return a dictionary comprising of data
            at dataset[ind].
        - __len__: return the size of the underlying dataset.
    '''
    DEFAULT_PARAMS = o(
        crop_params=MultiRandomAffineCrop.DEFAULT_PARAMS,
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
        self.annotation_path = "NA"
        self.img_dir = "NA"
        if train:
            self.multi_crop = MultiRandomAffineCrop(self.p.crop_params)
        else:
            self.multi_crop = MultiCenterAffineCrop(self.p.crop_params)
        self.encoder = SegEncoder(len(self.p.classes))
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=self.p.color_jitter.brightness,
            contrast=self.p.color_jitter.contrast,
            saturation=self.p.color_jitter.saturation,
            hue = self.p.color_jitter.hue
        )
    
    def augment_image(self, img):
        # TODO: add more augmentation
        if self.mode == 'train':
            img = self.color_jitter(img)
        return img
    
    def get_raw_data(self, key):
        raise NotImplementedError("Pure Virtual Method")

    def __getitem__(self, key):
        raw_data = self.get_raw_data(key)
        crop_data = self.multi_crop(raw_data)
        crop_data['image'] = self.augment_image(crop_data['image'])
        enc_data = self.encoder(crop_data)
        return enc_data

class COCODataset(BaseSet):
    DEFAULT_PARAMS = BaseSet.DEFAULT_PARAMS
    DEFAULT_PARAMS = DEFAULT_PARAMS(
        version=2017,
        data_dir='/data/COCO2017',
        annotation_dir='/data/COCO2017/annotations',
        min_area=200,
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
        ])
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

    def get_raw_data(self, ind):
        assert isinstance(ind, int), "non integer index not supported!"
        img_id = self.img_ids[ind]
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

    def __len__(self):
        return len(self.coco.imgs)

class ADE20KDataset(BaseSet):
    '''
    Fine-grained instance-level segmentation data from the 2016 ADE20K challenge.

    Data can be grabbed from http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    '''
    DEFAULT_PARAMS = BaseSet.DEFAULT_PARAMS(
        root_dir = "/data/ADEChallengeData2016/",
        classes=set([
            "wall",
            "floor, flooring",
            "ceiling",
            "bed",
            "cabinet",
            "door, double door",
            "table",
            "curtain, drape, drapery, mantle, pall",
            "chair",
            "sofa, couch, lounge",
            "shelf",
            "armchair",
            "seat",
            "desk",
            "lamp",
            "chest of drawers, chest, bureau, dresser",
            "pillow",
            "screen door, screen",
            "coffee table, cocktail table",
            "toilet, can, commode, crapper, pot, potty, stool, throne",
            "kitchen island",
            "computer, computing machine, computing device, data processor, electronic computer, information processing system",
            "swivel chair",
            "pole",
            "bannister, banister, balustrade, balusters, handrail",
            "cradle",
            "oven",
            "screen, silver screen, projection screen",
            "blanket, cover",
            "tray",
            "crt screen",
            "plate",
            "monitor, monitoring device"
        ])
    )

    def __init__(self, params=DEFAULT_PARAMS, train=True):
        '''
        Initialize and load the ADE20K annotation file into memory.

        Args:
            - root: path to the folder containing the ADE20K_2016_07_26 folder.
                e.g. It should be /data if images are in /data/ADE20K_2016_07_26/images
            - annFile: path to the serialized Matlab array file provided in the dataset.
                e.g. /data/ADE20K_2016_07_26/index_ade20k.mat
        '''
        super(ADE20KDataset, self).__init__(params, train)
        root_dir = self.p.root_dir
        if train:
            img_dir = os.path.join(root_dir, "images/training")
            seg_anno_path = os.path.join(root_dir, "annotations/training")
        else:
            img_dir = os.path.join(root_dir, "images/validation")
            seg_anno_path = os.path.join(root_dir, "annotations/validation")
        anno_path = os.path.join(root_dir, "sceneCategories.txt")
        class_desc_path = os.path.join(root_dir, "objectInfo150.txt")
        # Load file paths and annotations
        with open(anno_path) as f:
            anno_content = f.readlines()

        self.img_path_list = []
        self.scenario_list = []
        self.seg_path_list = []

        for line in anno_content:
            img_name, scene_name = line[:-1].split(' ') # remove eol
            if train and "val" in img_name:
                continue
            if not train and "train" in img_name:
                continue
            img_path = os.path.join(img_dir, img_name + '.jpg')
            seg_path = os.path.join(seg_anno_path, img_name + '.png')
            self.img_path_list.append(img_path)
            self.seg_path_list.append(seg_path)
            self.scenario_list.append(scene_name)

        assert len(self.img_path_list) == len(self.scenario_list) == len(self.seg_path_list)
        self.dataset_size = len(self.img_path_list)
        self.class_map = self._generate_class_map(class_desc_path)

    def get_raw_data(self, index):
        """
        Args:
            index (int): Index

        Returns:
            ret_dict
        """
        img_path = self.img_path_list[index]
        seg_path = self.seg_path_list[index]
        img = Image.open(img_path).convert('RGB')
        seg_mask = np.array(Image.open(seg_path), dtype = np.uint8)
        seg_mask = self.class_map(seg_mask)
        seg_mask = torch.tensor(seg_mask, dtype = torch.uint8)
        loss_mask = torch.zeros_like(seg_mask)
        return {'image': img, 'seg_mask': seg_mask, 'loss_mask': loss_mask}

    def _generate_class_map(self, class_desc_path):
        # Take subset of class
        with open(class_desc_path) as f:
            class_desc = f.readlines()

        class_desc = class_desc[1:] # Remove header
        class_desc = [line[:-1].split('\t') for line in class_desc] # remove eol
        class_name_list = [line[-1].strip(' ') for line in class_desc]
        
        map_dict = {}
        cur_idx = 1 # Background maps to 0
        for i in range(len(class_name_list)):
            n = class_name_list[i]
            if n in self.p.classes:
                # Original class id is i + 1.
                map_dict[i + 1] = cur_idx
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