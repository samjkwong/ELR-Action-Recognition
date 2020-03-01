import os
import random

root = '/vision/group/video/scratch/ucf101_old'
with open(os.path.join(root, 'trainlist01.txt'), 'rb') as f:
    lines = f.readlines()
random.shuffle(lines)
N_total = len(lines)
N_train = int(0.9*N_total)
print(N_train)
with open('train_split.txt', 'wb') as f:
    for i in range(N_train):
        f.write(lines[i])
with open('val_split.txt', 'wb') as f:
    for i in range(N_train, N_total):
        f.write(lines[i])
