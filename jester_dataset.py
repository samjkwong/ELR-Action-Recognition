import os
import csv
import glob
import numpy as np
import torch
import time

from PIL import Image
from torchvision.transforms import *

from collections import namedtuple

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']
ListDataJpeg = namedtuple('ListDataJpeg', ['id', 'label', 'path'])


def default_loader(path):
    return Image.open(path).convert('RGB')


class Jester_Dataset(torch.utils.data.Dataset):

    def __init__(self, root, split_file, labels, clip_size,
                 stride, is_val, transform=None, transform_hr=None,
                 loader=default_loader, is_kd=False):
        self.dataset_object = Jester_Parsing(split_file, labels, root)

        self.csv_data = self.dataset_object.csv_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform = transform
        self.transform_hr = transform_hr
        self.loader = loader

        self.clip_size = clip_size
        self.stride = stride 
        self.is_val = is_val
        self.is_kd = is_kd

    def __getitem__(self, index):
        item = self.csv_data[index]
        img_paths = self.get_frame_names(item.path)

        imgs = [] # could be either LR or HR images
        if self.is_kd: # need to get additional HR images for HR branch (in addition to imgs which holds LR)
            imgs_hr = []
        
        for img_path in img_paths:
            img = self.loader(img_path)
            img = self.transform(img)
            if self.is_kd:
                imgs_hr.append(torch.unsqueeze(self.transform_hr(img), 0))
            imgs.append(torch.unsqueeze(img, 0))

        target_idx = self.classes_dict[item.label]

        # format to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        if self.is_kd:
            data_hr = torch.cat(imgs_hr)
            data_hr = data_hr.permute(1, 0, 2, 3)
            return (data_hr, data, target_idx)
        else:
            return (data, target_idx)

    def __len__(self):
        return len(self.csv_data)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
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
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        frame_names = frame_names[offset:num_frames_necessary+offset:self.stride]
        return frame_names


class Jester_Parsing(object):

    def __init__(self, csv_path_input, csv_path_labels, data_root):
        self.csv_data = self.read_csv_input(csv_path_input, data_root)
        self.classes = self.read_csv_labels(csv_path_labels)
        self.classes_dict = self.get_two_way_dict(self.classes)

    def read_csv_input(self, csv_path, data_root):
        csv_data = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for row in csv_reader:
                item = ListDataJpeg(row[0],
                                    row[1],
                                    os.path.join(data_root, row[0])
                                    )
                csv_data.append(item)
        return csv_data

    def read_csv_labels(self, csv_path):
        classes = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                classes.append(row[0])
        return classes

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

