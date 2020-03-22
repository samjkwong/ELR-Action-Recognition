"""
* Adds class idx label to each line representing a video sample, to match train split file format
* Writes to ucf_test_split.txt
"""

import os

root = '/vision/group/video/scratch/ucf101_old'
with open(os.path.join(root, 'classInd.txt'), 'r') as f:
    class_to_idx = {} # maps class name to class idx
    for line in f:
        line = line.split()
        class_to_idx[line[1]] = line[0]
        
with open(os.path.join(root, 'testlist01.txt'), 'r') as f:
    lines = f.readlines()

if os.path.exists('ucf_test_split.txt'):
    os.remove('ucf_test_split.txt')
with open('ucf_test_split.txt', 'w') as f:
    for line in lines:
        f.write(line.strip('\n') + ' ' + class_to_idx[line.split('/')[0]] + '\n')
