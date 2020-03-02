"""
    * Shuffles training set directory names and splits 90%-10% train val
    * Writes to train_split.txt and val_split.txt
"""

import os
import random

root = '/vision/group/video/scratch/ucf101_old'
with open(os.path.join(root, 'trainlist01.txt'), 'r') as f:
    lines = f.readlines()
random.shuffle(lines)
N_total = len(lines)
N_train = int(0.9*N_total)

if os.path.exists('train_split.txt'):
    os.remove('train_split.txt')
with open('train_split.txt', 'w') as f:
    for i in range(N_train):
        f.write(lines[i])

if os.path.exists('val_split.txt'):
    os.remove('val_split.txt')
with open('val_split.txt', 'w') as f:
    for i in range(N_train, N_total):
        f.write(lines[i])
