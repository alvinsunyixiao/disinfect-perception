import math
import random

import torch
import torchvision

from utils.params import ParamDict as o

import numpy as np
import cv2
from PIL import Image

class AffineCrop:
    """ Rotated and scaled crop implemented with affine transformation """

    def __init__(self, output_hw, center_yx, rotation, scale_yx, interp=Image.NEAREST):
        self.output_hw = output_hw
        self.center_yx = center_yx
        self.rotation = rotation
        self.scale_yx = scale_yx
        self.interp = interp

    def get_affine_coeffs(self, output_hw, center_yx, rotation, scale_yx):
        rotation = math.radians(rotation)
        # top-left to center offset in output space
        offset_x = output_hw[1] / 2
        offset_y = output_hw[0] / 2
        # pre-calculate trignometric values
        cos_theta = math.cos(rotation)
        sin_theta = math.sin(rotation)
        # calculate affine parameters
        a = cos_theta / scale_yx[1]
        b = sin_theta / scale_yx[0]
        c = center_yx[1] - a * offset_x - b * offset_y
        d = -sin_theta / scale_yx[1]
        e = cos_theta / scale_yx[0]
        f = center_yx[0] - d * offset_x - e * offset_y
        return (a, b, c, d, e, f)

    def __call__(self, pil_img):
        assert isinstance(pil_img, Image.Image), "Not a PIL image!"
        affine_coeffs = self.get_affine_coeffs(self.output_hw, self.center_yx,
                                               self.rotation, self.scale_yx)
        return pil_img.transform(self.output_hw[::-1], Image.AFFINE,
                                 affine_coeffs, resample=self.interp)

class RandomAffineCrop(AffineCrop):
    """ Randomly rotated and scaled crop implemented with affine transformation """

    DEFAULT_PARAMS=o(
        # output image size
        output_hw=(256, 256),
        # maximum absolute rotation in [deg]
        max_abs_rotation=30,
        # zoom scale, the bigger the number, the more zoomed in
        min_log_scale=math.log(.2),
        max_log_scale=math.log(1.5),
        # aspect ratio defined as scale_y / scale_x
        min_log_aspect_ratio=math.log(3/4),
        max_log_aspect_ratio=math.log(4/3),
        # random flip
        flip_lr=True, # left / right
        flip_ud=False, # up / down
        # interpolation used for transformation
        interp=Image.NEAREST,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

    def get_random_affine_crop(self, input_hw):
        # generate randomly scaled aspect ratio
        scale = math.exp(random.uniform(self.p.min_log_scale, self.p.max_log_scale))
        area_scale = scale**2
        aspect_ratio = math.exp(random.uniform(self.p.min_log_aspect_ratio,
                                               self.p.max_log_aspect_ratio))
        scale_y = math.sqrt(area_scale/aspect_ratio)
        scale_x = area_scale / scale_y
        # clip the scaling if it exceeds input image size
        ratio_y = self.p.output_hw[0] / input_hw[0] / scale_y
        ratio_x = self.p.output_hw[1] / input_hw[1] / scale_x
        max_ratio = max(ratio_y, ratio_x)
        if max_ratio > 1:
            scale_y *= max_ratio
            scale_x *= max_ratio
        # get crop size in input image space
        crop_h = self.p.output_hw[0] / scale_y
        crop_w = self.p.output_hw[1] / scale_x
        # generate crop center that ensures the crop lies within the input image
        center_y = random.uniform(crop_h / 2, input_hw[0] - crop_h / 2)
        center_x = random.uniform(crop_w / 2, input_hw[1] - crop_w / 2)
        # random flips
        if self.p.flip_lr:
            scale_x *= random.choice([-1, 1])
        if self.p.flip_ud:
            scale_y *= random.choice([-1, 1])
        # generate random rotations
        rotation = random.uniform(-self.p.max_abs_rotation, self.p.max_abs_rotation)
        return (center_y, center_x), rotation, (scale_y, scale_x)

    def __call__(self, pil_img):
        center_yx, rotation, scale_yx = self.get_random_affine_crop(pil_img.size[::-1])
        affine_coeffs = self.get_affine_coeffs(
            self.p.output_hw, center_yx, rotation, scale_yx)
        return pil_img.transform(self.p.output_hw[::-1], Image.AFFINE,
                                 affine_coeffs, resample=self.p.interp)

class MultiRandomAffineCrop(RandomAffineCrop):
    """ apply the same random affine crop to multiple inputs """

    DEFAULT_PARAMS=RandomAffineCrop.DEFAULT_PARAMS

    def __init__(self, params=DEFAULT_PARAMS):
        super(MultiRandomAffineCrop, self).__init__(params)
        self.tensor_to_pil = torchvision.transforms.ToPILImage()

    def __call__(self, input_dict, keys=None):
        assert isinstance(input_dict, dict), "Input must be a dictionary"
        if keys is None:
            keys = list(input_dict.keys())
        assert isinstance(keys, list), "Keys must be a list"
        # do not modify input
        output_dict = input_dict.copy()
        # get random transforms
        if isinstance(output_dict[keys[0]], torch.Tensor):
            output_dict[keys[0]] = self.tensor_to_pil(output_dict[keys[0]])
        input_hw = output_dict[keys[0]].size[::-1]
        center_yx, rotation, scale_yx = self.get_random_affine_crop(input_hw)
        affine_coeffs = self.get_affine_coeffs(
            self.p.output_hw, center_yx, rotation, scale_yx)
        for key in keys:
            if output_dict[key] is None:
                continue
            if isinstance(output_dict[key], list):
                continue # meta data
            if isinstance(output_dict[key], torch.Tensor):
                output_dict[key] = self.tensor_to_pil(output_dict[key])
            assert output_dict[key].size[::-1] == input_hw, \
                "Size mismatch: {} vs {}".format(input_hw, output_dict[key].size[::-1])
            output_dict[key] = output_dict[key].transform(
                self.p.output_hw[::-1], Image.AFFINE, affine_coeffs, resample=self.p.interp)
        return output_dict

class MultiCenterAffineCrop(MultiRandomAffineCrop):

    DEFAULT_PARAMS=MultiRandomAffineCrop.DEFAULT_PARAMS

    def __init__(self, params=DEFAULT_PARAMS):
        super(MultiCenterAffineCrop, self).__init__(params)

    def get_random_affine_crop(self, input_hw):
        center_y = input_hw[0] / 2
        center_x = input_hw[1] / 2
        rotation = 0
        scale_y = self.p.output_hw[0] / input_hw[0]
        scale_x = self.p.output_hw[1] / input_hw[1]
        max_scale = max(scale_y, scale_x)
        return (center_y, center_x), rotation, (max_scale, max_scale)

class AdditiveGaussianNoise:

    def __init__(self, std, prob=0.8, per_channel=0.5):
        self.std = std
        self.prob = prob
        self.per_channel = per_channel

    def __call__(self, img_3hw):
        if random.random() < self.prob:
            if random.random() < self.per_channel:
                noise = torch.randn(img_3hw.shape) * self.std
            else:
                noise = torch.randn((1,) + img_3hw.shape[1:]) * self.std
            return img_3hw + noise
        else:
            return img_3hw

class RandomColorJitter:

    def __init__(self, brightness, contrast, saturation, hue, prob=0.8):
        self.prob = prob
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img_pil):
        assert isinstance(img_pil, Image.Image), "input must be PIL image"
        if random.random() < self.prob:
            return self.color_jitter(img_pil)
        else:
            return img_pil

class RandomMotionBlur:

    def __init__(self, prob = 0.1, size_range = [3, 5, 10, 15]):
        self.prob = prob
        self.size_range = size_range
        self.blur_dir_pool = (
            self.horizontal_motion_blur,
            self.vertical_motion_blur,
            self.diagonal_motion_blur,
            self.inverse_diagonal_motion_blur
        )

    def __call__(self, img_3hw):
        if random.random() < self.prob:
            img_3hw = img_3hw.numpy().transpose((1, 2, 0))
            blur_kernel_size = random.choice(self.size_range)
            blur_dir_func = random.choice(self.blur_dir_pool)
            img_3hw = blur_dir_func(img_3hw, blur_kernel_size)
            img_3hw = img_3hw.transpose((2, 0, 1))
            img_3hw = torch.tensor(img_3hw)
            return img_3hw
        else:
            return img_3hw
    
    @staticmethod
    def horizontal_motion_blur(img, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2),:] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(img, -1, kernel) 

    @staticmethod
    def vertical_motion_blur(img, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:,int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(img, -1, kernel) 

    @staticmethod
    def diagonal_motion_blur(img, kernel_size):
        kernel = np.eye(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def inverse_diagonal_motion_blur(img, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            kernel[i,kernel_size - i - 1] = 1
        kernel /= kernel_size
        return cv2.filter2D(img, -1, kernel)

class ImageAugmentor:

    DEFAULT_PARAMS=o(
        color_jitter=o(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            prob=0.8,
        ),
        random_motion_blur=o(
            prob = 0.1221,
            size_range = (3, 5, 10 ,15)
        ),
        gauss_noise=o(
            std=0.05,
            prob=0.8,
            per_channel=0.5,
        ),
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params
        self.pil_to_tensor = torchvision.transforms.ToTensor()
        self.color_jitter = RandomColorJitter(
            brightness=self.p.color_jitter.brightness,
            contrast=self.p.color_jitter.contrast,
            saturation=self.p.color_jitter.saturation,
            hue=self.p.color_jitter.hue,
            prob=self.p.color_jitter.prob,
        )
        self.random_motion_blur = RandomMotionBlur(
            prob=self.p.random_motion_blur.prob,
            size_range=self.p.random_motion_blur.size_range
        )
        self.gauss_noise = AdditiveGaussianNoise(
            std=self.p.gauss_noise.std,
            prob=self.p.gauss_noise.prob,
            per_channel=self.p.gauss_noise.per_channel,
        )

    def __call__(self, img_pil):
        img_pil = self.color_jitter(img_pil)
        img = self.pil_to_tensor(img_pil)
        img = self.random_motion_blur(img)
        img = self.gauss_noise(img)
        img = torch.clamp(img, 0, 1)
        return img
