#!/usr/bin/env python3
"""
* Contributor: Samuel Kwong
* Example usage: python eval_i3d.py --bs=16 --stride=1 --clip_size=250 --num-workers=4 --checkpoint_path=models/baseline-ucf.pt
"""

import os
import sys
import argparse
import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

import numpy as np
from pytorch_i3d import InceptionI3d
from ucf_dataset import UCF_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--stride', type=int, help='temporal stride for sampling input frames')
parser.add_argument('--clip_size', type=int, help='total number of frames to sample per input for testing')
parser.add_argument('--num_workers', type=int, help='number of cpu threads for dataloader')
parser.add_argument('--checkpoint_path', type=str, help='path to saved checkpoint (\'\' to test from kinetics baseline)')
args = parser.parse_args()


def test(root, batch_size, stride, clip_size, test_split, num_workers):
    dataloader = get_dataloader(root, stride, clip_size, batch_size, test_split, num_workers) 
    
    # ----------------------- LOAD MODEL ---------------------------
    print('Loading model...')
    i3d = InceptionI3d(400, in_channels=3)
    if args.checkpoint_path:
        i3d.replace_logits(101)
        state_dict = torch.load(args.checkpoint_path)['model_state_dict']
        checkpoint = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module'
            checkpoint[name] = v
        i3d.load_state_dict(checkpoint)
    else:
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        i3d.replace_logits(101)

    i3d.cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        i3d = nn.DataParallel(i3d)
    i3d.to(device)
    print('Loaded model.')
    
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)

    # ----------------------------------------------------------------
    
    # ---------------------------- TEST ------------------------------
    epoch_start_time = time.time()

    i3d.train(False)  # set model to evaluate mode
    print('-'*10, 'TESTING', '-'*10)
    
    print('Entering data loading...')
    steps = 0
    num_correct = 0
    step_start_time = time.time()
    for data in dataloader:
        inputs, labels = data
        t = inputs.shape[2]
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        with torch.no_grad():
            per_frame_logits = i3d(inputs)

        # upsample to input size
        per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear') # shape: B x Classes x T
        mean_frame_logits = torch.mean(per_frame_logits, dim=2) # shape: B x Classes; avg across frames to get single pred per clip
        _, pred_class_idx = torch.max(mean_frame_logits, dim=1) # shape: B; values are class indices

        print('Step {} {}s'.format(steps, time.time()-step_start_time))
            
        # metrics for validation
        num_correct += torch.sum(pred_class_idx == labels, axis=0)
        steps += 1
        step_start_time = time.time()

    # ----------------------- EVALUATE ACCURACY ---------------------------
    num_total = len(dataloader.dataset)
    accuracy = float(num_correct) / float(num_total)
    elapsed_time = time.time() - epoch_start_time
    print('Elapsed time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    print('Testing accuracy: {:.4f}'.format(accuracy))
    # --------------------------------------------------------------------

    # ---------------------------------------------------------------------------------- 

# ------------------------------------- HELPERS ------------------------------------------
def get_dataloader(root, stride, clip_size, batch_size, test_split, num_workers):
    print('Getting testing dataset...')
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()
                                         ])
    test_dataset = UCF_Dataset(root, split_file=test_split, clip_size=clip_size, stride=stride, is_val=True, transform=test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_dataloader

# --------------------------------------------------------------------------------------


if __name__ == '__main__':
    if len(sys.argv) < len(vars(args))+1:
        parser.print_usage()
        parser.print_help()
    else:
        print('Starting...')
        now = datetime.datetime.now()

        BATCH_SIZE = args.bs
        STRIDE = args.stride # temporal stride for sampling
        CLIP_SIZE = args.clip_size # total number frames to sample for inputs
        NUM_WORKERS = args.num_workers # num cpu threads for dataloader

        test(root='/vision/group/video/scratch/ucf101_old/two_stream_flow_frame', batch_size=BATCH_SIZE,
             stride=STRIDE, clip_size=CLIP_SIZE, test_split='ucf_test_split.txt', num_workers=NUM_WORKERS)

