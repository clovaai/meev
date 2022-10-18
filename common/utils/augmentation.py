"""
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License
"""


import numpy as np
import random
import albumentations as A



from utils.cfg_utils import getIntFromCfg


def augmentation_method(cfg, data_split):
    if data_split == 'train':
        aug_type = getIntFromCfg(cfg, 'augmentation', 0)
        if aug_type == -1:
            return no_augmentation
        if aug_type == 0:
            return augmentation_0
        if aug_type == 1:
            return augmentation_1
        if aug_type == 2:
            return augmentation_2
    
    return None

def no_augmentation(img):
    return img

def augmentation_0(img):
    color_factor = 0.2
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img

t1 = A.Compose([
    A.OneOf([
        A.GaussNoise(),
        A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        A.MultiplicativeNoise( multiplier=(0.8, 1.2), per_channel=True, elementwise=True),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),            
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])

t2 = A.Compose([
    A.OneOf([
        A.CoarseDropout(max_holes=10, max_height=20, max_width=20, min_holes=2, min_height=2, min_width=2)
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(),
        A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        A.MultiplicativeNoise( multiplier=(0.8, 1.2), per_channel=True, elementwise=True),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),            
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])

def augmentation_1(img):
    img = t1(image=img.astype(np.uint8))
    return img['image']

def augmentation_2(img):
    img = t2(image=img.astype(np.uint8))
    return img['image']