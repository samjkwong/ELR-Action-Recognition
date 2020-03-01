import os
import glob
import numpy as np
import torch

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
    def __init__(self, root, split_file, clip_size, stride, is_val=False, transform=None, loader=default_loader):
        self.dataset = make_dataset(split_file)
        self.root = root
        self.loader = loader
        self.clip_size = clip_size
        self.stride = stride 
        self.is_val = is_val
        self.transform = transform

    def __getitem__(self, index):
        path, action_idx = self.dataset[index]
        img_paths = self.get_frame_names(self.root + '/' + path) # where img dirs are stored

        imgs = []
        for img_path in img_paths:
            img = self.loader(img_path)
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        return (data, action_idx)

    def __len__(self):
        return len(self.dataset)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(path + "/image_[0-9]*" + ext))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        # set number of necessary frames
        #if not self.is_val:
        num_frames_necessary = self.clip_size * self.stride
        #else:
        #    num_frames_necessary = num_frames # only works with batch size 1

        # pick frames
        
        # random offset sampling
        #offset = 0
        #if num_frames_necessary > num_frames:
        #    # Pad last frame if video is shorter than necessary
        #    frame_names += [frame_names[-1]] * \
        #        (num_frames_necessary - num_frames)
        #elif num_frames_necessary < num_frames:
        #    # If there are more frames, then sample starting offset
        #    diff = (num_frames - num_frames_necessary)
        #    # temporal augmentation
        #    if not self.is_val:
        #        offset = np.random.randint(0, diff)
        #frame_names = frame_names[offset:num_frames_necessary+offset:self.stride]
        
        # middle of video offset sampling for both train and val
        offset = 0
        if num_frames_necessary > num_frames:
            # Pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * \
                (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset
            diff = (num_frames - num_frames_necessary)
            offset = diff // 2
        frame_names = frame_names[offset:num_frames_necessary+offset:self.stride]
        return frame_names

