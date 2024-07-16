from __future__ import print_function
import os
import cv2
import numpy as np
import random
import torch
from torch.utils import data
from torchvision import transforms
import PIL.Image

def read_list(list_file):
    # rgb_depth_list = []
    # with open(list_file) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         rgb_depth_list.append(line.strip().split(" "))
    # return rgb_depth_list
    file_names = []
    with open(list_file) as f:
        files = f.readlines()

    for item in files:
        file_name = item.strip()
        file_names.append(file_name)

    return file_names

class Struct3D(data.Dataset):
    NUM_CLASSES = 40
    ID2CLASS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
                'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub',
                'bag', 'otherstructure', 'otherfurniture', 'otherprop']
    def __init__(self, root_dir, list_file, hw,
                 flip=False, rotate=False, is_training=False, mask_black = False):
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        self.h, self.w = hw
        self.area = 'full'
        self.lighting = 'rawlight'
        self.max_depth_meters = 8.0
        self.mask_black = mask_black
        self.LR_filp_augmentation = not flip
        self.yaw_rotation_augmentation = not rotate

        self.is_training = is_training
        self.ignore_index = 255
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        item_area, item_name = self.rgb_depth_list[idx].strip().split(' ')
        rgb_name = os.path.join(self.root_dir, item_area, '2D_rendering', item_name, 'panorama', self.area,f'rgb_{self.lighting}.png')
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        rgb = torch.FloatTensor(rgb / 255.).permute(2, 0, 1)
        ##################depth
        # depth_name = os.path.join(self.root_dir, item_area, '2D_rendering', item_name, 'panorama', self.area,'depth.png')
        # gt_depth = cv2.imread(depth_name, -1)
        # # gt_depth = cv2.imread(depth_name, cv2.IMREAD_GRAYSCALE)
        # gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        # gt_depth = gt_depth.astype(np.float64)/500.
        # gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        sem_path = os.path.join(self.root_dir, item_area, '2D_rendering', item_name, 'panorama', self.area,
                                'semantic.png')
        sem = PIL.Image.open(sem_path).resize((1024, 512), PIL.Image.NEAREST)
        sem = np.asarray(sem, dtype=np.uint8)
        sem = sem - 1
        inputs["sem"] = sem
        inputs["sem"][inputs["sem"] == 255] = self.ignore_index  # mask as unknown id: 255
        inputs["sem"] =torch.LongTensor(inputs["sem"])
        if self.is_training and self.yaw_rotation_augmentation:
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            # gt_depth = np.roll(gt_depth, roll_idx, 1)
            sem = np.roll(sem, roll_idx, 1)
        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            # gt_depth = cv2.flip(gt_depth, 1)
            sem = cv2.flip(sem, 1)
        aug_rgb = rgb
        inputs["sem_path"] = [item_area,item_name,self.area]
        inputs["ori_path"] = sem_path
        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)


        return inputs



