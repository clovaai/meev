"""
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import os.path as osp
import random
import torch
import numpy as np

from utils.cfg_utils import getIntFromCfg
from utils.preprocessing import get_aug_config


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, data_split):
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split

        self.visualize_info = False
            

    def is_train(self):
        return self.data_split == 'train'

    def is_body_parts(self):
        return self.cfg.parts == 'body'

    def is_hand_parts(self):
        return self.cfg.parts == 'hand'

    def is_face_parts(self):
        return self.cfg.parts == 'face'

    def get_aug_config(self):
        if self.is_train() and getIntFromCfg(self.cfg, 'augmentation', 0) != -1:
            aug_config = get_aug_config(self.cfg)
        else:
            aug_config = 1.0, 0.0, np.array([1,1,1]), False, 0.0, 0.0
        return aug_config

