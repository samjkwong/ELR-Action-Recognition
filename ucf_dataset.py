import os
import glob
import numpy as np
import torch
import pdb

from PIL import Image
from torchvision.transforms import *

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']


def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(split_file):
    dataset = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.split()
            path, action_idx = line[0], line[1]
            dataset.append((path, int(action_idx) - 1))
    return dataset

class UCF_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split_file, clip_size, stride,
                 is_test, transform=None, transform_hr=None,
                 loader=default_loader, is_kd=False):
        self.dataset = make_dataset(split_file)
        self.root = root
        self.loader = loader
        self.clip_size = clip_size
        self.stride = stride 
        self.is_test = is_test
        self.transform = transform
        self.is_kd = is_kd
        self.empty_img = self.transform(self.loader('/vision2/u/samkwong/ELR-Action-Recognition/empty.jpg'))

    def __getitem__(self, index):
        path, action_idx = self.dataset[index]
        img_paths = self.get_frame_names(self.root + '/' + path) # where img dirs are stored

        imgs = [] # could be either LR or HR images
        if self.is_kd:
            imgs_hr = []
            prev_img_hr = self.empty_img

        prev_img = self.empty_img
        for img_path in img_paths:
            try:
                img = self.loader(img_path)
                img = self.transform(img)
                prev_img = img
                if self.is_kd:
                    img_hr = self.loader(img_path)
                    img_hr = self.transform_hr(img_hr)
                    imgs_hr.append(torch.unsqueeze(img_hr, 0))
                    prev_img_hr = img_hr
            except: # corrupted/empty file
                img = prev_img
                img_hr = prev_img_hr
            imgs.append(torch.unsqueeze(img, 0))

        # format to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        if self.is_kd:
            data_hr = torch.cat(imgs_hr)
            data_hr = data_hr.permute(1, 0, 2, 3)
            return (data_hr, data, action_idx)
        else:
            return (data, action_idx)

    def __len__(self):
        return len(self.dataset)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(path + "/image_[0-9]*" + ext))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        num_frames_necessary = self.clip_size * self.stride
        
        offset = 0
        if num_frames_necessary > num_frames:
            # pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * \
                (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # if there are more frames, sample starting offset
            diff = (num_frames - num_frames_necessary)
            if not self.is_test:
                offset = np.random.randint(0, diff) # temporal augmentation
            else:
                offset = diff // 2 # center sample for testing
        frame_names = frame_names[offset:num_frames_necessary+offset:self.stride]
        
        return frame_names

