import os
import glob
import numpy as np
from imageio import imread
import cv2
import json
from torchvision import transforms
import torch
import torch.utils.data as data
import torch.nn.functional as F

# from lib.misc import panostretch

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_valid': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_valid': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_valid': ['area_1', 'area_3', 'area_6'],
}
with open('G:/Stanford2D3D_sem/semantic_labels.json') as f:
    id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
with open('G:/Stanford2D3D_sem/name2label.json') as f2:
    name2id = json.load(f2)

class S2d3dSemgDataset(data.Dataset):
    NUM_CLASSES = 13
    ID2CLASS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
    def __init__(self, root, fold, depth=False, hw=(512, 1024), mask_black=False, flip=False, rotate=False):
        assert fold in __FOLD__, 'Unknown fold'
        self.depth = depth
        self.h, self.w = hw
        self.mask_black = mask_black
        self.rgb_paths = []
        self.sem_paths = []
        self.dep_paths = []
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)

        for dname in __FOLD__[fold]:
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'pano', 'rgb', '*png'))))
            self.sem_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'pano', 'semantic', '*png'))))
            self.dep_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'pano', 'depth', '*png'))))
        assert len(self.rgb_paths)
        assert len(self.rgb_paths) == len(self.sem_paths)
        assert len(self.rgb_paths) == len(self.dep_paths)
        self.flip = flip
        self.rotate = rotate
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = cv2.imread(self.rgb_paths[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        rgb = torch.FloatTensor(rgb / 255.).permute(2, 0, 1)

        sem = cv2.imread(self.sem_paths[idx])
        sem = cv2.cvtColor(sem, cv2.COLOR_BGR2RGB)
        sem = cv2.resize(sem, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem = torch.LongTensor(sem) - 1

        if self.depth:
            dep = cv2.imread(self.dep_paths[idx], -1)
            dep = cv2.resize(dep, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
            dep = np.where(dep==65535, 0, dep/512)
            dep = np.clip(dep, 0, 4)
            dep = torch.FloatTensor(dep[None])
            rgb = torch.cat([rgb, dep], 0)
        # Random flip
        if self.flip and np.random.randint(2) == 0:
            rgb = torch.flip(rgb, (-1,))
            sem = torch.flip(sem, (-1,))

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(self.w)
            rgb = torch.roll(rgb, dx, dims=-1)
            sem = torch.roll(sem, dx, dims=-1)

        # Mask out top-down black
        if self.mask_black:
            sem[rgb.sum(0) == 0] = -1
        rgb_true = rgb
        rgb = self.normalize(rgb)

        # Convert all data to tensor
        out_dict = {
            'x': rgb,
            'rgb': rgb_true,
            'sem': sem,
            'fname': os.path.split(self.rgb_paths[idx])[1].ljust(200),
        }
        return out_dict

