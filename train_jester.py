#!/usr/bin/env python3

"""
* Contributor: Samuel Kwong
* Script to finetune: 
    0) low-resolution (LR) baseline on Jester-ELR,
    1) high-resolution (HR) branch on Jester, or
    2) cross-resolution knowledge distillation (KD) model on Jester-ELR
* Backbone model: I3D
* Dataset: Jester-ELR
* LR baseline example usage:
    python train_jester.py --lr=1e-4 --bs=16 --stride=1 --clip_size=32 --model_num=0 --ckpt_hr='' --ckpt_lr=''
* HR (high-resolution) branch example usage:
    python train_jester.py --lr=1e-4 --bs=16 --stride=1 --clip_size=32 --model_num=1 --ckpt_hr='' --ckpt_lr=''
* KD (knowledge distillation) model example usage:
    python train_jester.py --lr=1e-4 --bs=16 --stride=1 --clip_size=32 --model_num=2 --ckpt_hr=models/hr-branch.pt --ckpt_lr=''
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
from jester_dataset import Jester_Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--stride', type=int, help='temporal stride for sampling input frames (choose 1 for jester)')
parser.add_argument('--clip_size', type=int, help='total number of frames to sample per input for training')
parser.add_argument('--model_num', type=int, help='which model to finetune (0=LR, 1=HR, 2=KD)')
parser.add_argument('--ckpt_hr', type=str, help='path to saved high-res model checkpoint (\'\' to train from kinetics baseline)')
parser.add_argument('--ckpt_lr', type=str, help='path to saved low-res model checkpoint (\'\' to train from kinetics baseline)')
args = parser.parse_args()


def train(model_num, init_lr, batch_size, stride, clip_size, num_epochs, save_dir, root, train_split, val_split, labels):
    writer = SummaryWriter()
    dataloaders = get_dataloaders(model_num, root, stride, clip_size, batch_size, train_split, val_split, labels) 
    
    # ----------------------- LOAD MODEL ---------------------------
    if model_num != 0:
        print('Loading HR model...')
        i3d_hr = InceptionI3d(400, in_channels=3)
        if args.ckpt_hr:
            i3d_hr.replace_logits(101)
            state_dict = torch.load(args.ckpt_hr)['model_state_dict']
            checkpoint = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module'
                checkpoint[name] = v
            i3d_hr.load_state_dict(checkpoint)
        else:
            i3d_hr.load_state_dict(torch.load('models/rgb_imagenet.pt'))
            i3d_hr.replace_logits(101)
        i3d_hr.cuda()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            i3d_hr = nn.DataParallel(i3d_hr)
        i3d_hr.to(device)

    if model_num != 1:
        print('Loading LR model...')
        i3d_lr = InceptionI3d(400, in_channels=3)
        if args.ckpt_lr:
            i3d_hr.replace_logits(101)
            state_dict = torch.load(args.ckpt_lr)['model_state_dict']
            checkpoint = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module'
                checkpoint[name] = v
            i3d_lr.load_state_dict(checkpoint)
        else:
            i3d_lr.load_state_dict(torch.load('models/rgb_imagenet.pt'))
            i3d_lr.replace_logits(101)
        i3d_lr.cuda()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            i3d_lr = nn.DataParallel(i3d_lr)
        i3d_lr.to(device)
     
    if model_num == 0: 
        optimizer = optim.Adam(i3d_lr.parameters(), lr=init_lr)
    elif model_num == 1:
        optimizer = optim.Adam(i3d_hr.parameters(), lr=init_lr)
    else:
        optimizer = optim.Adam(list(i3d_hr.parameters()) + list(i3d_lr.parameters()), lr=init_lr)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)

    steps = 1
    start_epoch = 1
    best_epoch, best_acc = 1, 0.0
    if model_num != 1 and args.ckpt_lr:
        steps = torch.load(args.ckpt_lr)['steps']
        start_epoch = torch.load(args.ckpt_lr)['epoch']
        best_epoch, best_acc = torch.load(args.ckpt_lr)['best']
    if model_num == 1 and args.ckpt_hr:
        steps = torch.load(args.ckpt_hr)['steps']
        start_epoch = torch.load(args.ckpt_hr)['epoch']
        best_epoch, best_acc = torch.load(args.ckpt_hr)['best']

    # ----------------------------------------------------------------
    
    # ------------------------- TRAIN ------------------------------
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        print('-' * 50)
        print('EPOCH {}/{}'.format(epoch, num_epochs))
        print('-' * 50)

        for phase in ['train', 'val']:
            if phase == 'train':
                if model_num == 0:
                    i3d_lr.train(True)
                elif model_num == 1:
                    i3d_hr.train(True)
                else:
                    i3d_hr.train(False) # freeze pre-trained HR branch
                    i3d_lr.train(True)
                print('-'*10, 'TRAINING', '-'*10)
            else:
                if model_num == 0:
                    i3d_lr.train(False)
                elif model_num == 1:
                    i3d_hr.train(False)
                else:
                    i3d_hr.train(False)
                    i3d_lr.train(False)
                print('-'*10, 'VALIDATION', '-'*10)
            
            print('Entering data loading...')
            num_correct = 0
            for i, data in enumerate(dataloaders[phase]):
                if model_num == 0:
                    inputs, labels = data
                    inputs = inputs.to(device=device)
                    labels = labels.to(device=device)
                    if phase == 'train':
                        per_frame_logits = i3d_lr(inputs)
                    else:
                        with torch.no_grad():
                            per_frame_logits = i3d_lr(inputs)
                if model_num == 1:
                    inputs, labels = data
                    inputs = inputs.to(device=device)
                    labels = labels.to(device=device)
                    if phase == 'train':
                        per_frame_logits = i3d_hr(inputs)
                    else:
                        with torch.no_grad():
                            per_frame_logits = i3d_hr(inputs)
                if model_num == 2:
                    inputs_hr, inputs_lr, labels = data
                    inputs_hr = inputs_hr.to(device=device)
                    inputs_lr = inputs_lr.to(device=device)
                    labels = labels.to(device=device)
                    if phase == 'train':
                        with torch.no_grad():
                            per_frame_logits_hr = i3d_hr(inputs_hr)
                        per_frame_logits_lr = i3d_lr(inputs_lr)
                    else:
                        with torch.no_grad():
                            per_frame_logits_hr = i3d_hr(inputs_hr)
                            per_frame_logits_lr = i3d_lr(inputs_lr)
                

                if model_num < 2:
                    mean_frame_logits = torch.mean(per_frame_logits, dim=2) # shape: B x Classes; avg across frames to get single pred per clip
                    _, pred_class_idx = torch.max(mean_frame_logits, dim=1) # shape: B; values are class indices
                else:
                    mean_frame_logits_hr = torch.mean(per_frame_logits_hr, dim=2)
                    _, pred_class_idx_hr = torch.max(mean_frame_logits_hr, dim=1)
                    mean_frame_logits_lr = torch.mean(per_frame_logits_lr, dim=2)
                    _, pred_class_idx_lr = torch.max(mean_frame_logits_lr, dim=1)

                if phase == 'train':
                    if model_num < 2:
                        loss = F.cross_entropy(mean_frame_logits, labels)
                    else:
                        hr_ce_loss = F.cross_entropy(mean_frame_logits_hr, labels)
                        lr_ce_loss = F.cross_entropy(mean_frame_logits_lr, labels)
                        mean_frame_logits_hr = torch.nn.LogSoftmax(dim=1)(mean_frame_logits_hr)
                        mean_frame_logits_lr = torch.nn.Softmax(dim=1)(mean_frame_logits_lr)
                        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(mean_frame_logits_hr, mean_frame_logits_lr)
                        c1, c2 = 1, 1
                        loss = c1*lr_ce_loss + c2*kl_loss
                    writer.add_scalar('loss/train', loss, steps)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if steps % 10 == 0:
                        print('epoch {} step {} loss: {:.4f}'.format(epoch, steps, loss))
                    steps += 1
                    
                # metrics for validation
                num_correct += torch.sum(pred_class_idx == labels, axis=0)
        
            # ----------------------- EVALUATE ACCURACY ---------------------------
            num_total = len(dataloaders[phase].dataset)
            accuracy = float(num_correct) / float(num_total)
            elapsed_time = time.time() - start_time
            if phase == 'train':
                writer.add_scalar('accuracy/train', accuracy, epoch)
                print('-' * 50)
                print('{} accuracy: {:.4f}'.format(phase, accuracy))
                print('-' * 50)
                if model_num == 0 or model_num == 2:
                    save_checkpoint(i3d_lr, optimizer, loss, save_dir, epoch, steps, best_epoch, best_acc) # save checkpoint after epoch!
                else:
                    save_checkpoint(i3d_hr, optimizer, loss, save_dir, epoch, steps, best_epoch, best_acc) # save checkpoint after epoch!
                print('epoch {} elapsed time: {}'.format(epoch, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            else:
                writer.add_scalar('accuracy/val', accuracy, epoch)
                if accuracy > best_acc:
                    best_epoch = epoch
                    best_acc = accuracy
                print('{} accuracy: {:.4f}'.format(phase, accuracy))
                print('best accuracy: epoch {}: {}'.format(best_epoch, best_acc)) 
                print('val elapsed time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            # --------------------------------------------------------------------
        
        #lr_sched.step() # step after epoch
        
    writer.close()
    # ---------------------------------------------------------------------------------- 

# ------------------------------------- HELPERS ------------------------------------------
def get_dataloaders(model_num, root, stride, clip_size, batch_size, train_split, val_split, labels):
    print('Getting training dataset...')
    num_workers = 8

    def single(trans):
        train_dataset = Jester_Dataset(root, split_file=train_split, labels=labels,
                                       clip_size=clip_size, stride=stride, is_val=False, transform=trans)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=num_workers, pin_memory=True)
        print('Getting validation dataset...')
        
        val_dataset = Jester_Dataset(root, split_file=val_split, labels=labels,
                                     clip_size=clip_size, stride=stride, is_val=True, transform=trans)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                     shuffle=False, num_workers=num_workers, pin_memory=True)    

        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        return dataloaders
    
    def pair(trans_hr, trans_lr):
        train_dataset = Jester_Dataset(root, split_file=train_split, labels=labels,
                                       clip_size=clip_size, stride=stride, is_val=False, transform=trans_lr, transform_hr=trans_hr, is_kd=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=num_workers, pin_memory=True)
        print('Getting validation dataset...')
        
        val_dataset = Jester_Dataset(root, split_file=val_split, labels=labels,
                                     clip_size=clip_size, stride=stride, is_val=True, transform=trans_lr, transform_hr=trans_hr, is_kd=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                     shuffle=False, num_workers=num_workers, pin_memory=True)    

        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        return dataloaders

    trans_lr = transforms.Compose([transforms.Resize((12,16)),
                                transforms.Resize((224,224)),
                                transforms.ToTensor()
                               ])
    trans_hr = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()
                               ])
    if model_num == 0:
        dataloaders = single(trans_lr)
    elif model_num == 1:
        dataloaders = single(trans_hr)
    else:
        dataloaders = pair(trans_hr=trans_hr, trans_lr=trans_lr)
    
    return dataloaders


def save_checkpoint(model, optimizer, loss, save_dir, epoch, steps, best_epoch, best_acc):
    """Saves checkpoint of model weights during training."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = save_dir + str(epoch).zfill(3) + '.pt'
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'steps': steps,
                'best': (best_epoch, best_acc)
                },
                save_path)
# --------------------------------------------------------------------------------------


if __name__ == '__main__':
    if len(sys.argv) < len(vars(args))+1:
        parser.print_usage()
        parser.print_help()
    else:
        print('Using {} GPU(s)'.format(torch.cuda.device_count()))
        print('Starting...')
        now = datetime.datetime.now()

        MODEL_NUM = args.model_num
        LR = args.lr
        BATCH_SIZE = args.bs
        STRIDE = args.stride # temporal stride for sampling
        CLIP_SIZE = args.clip_size # total number frames to sample for inputs
        NUM_EPOCHS = 50
        SAVE_DIR = './checkpoints-{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}/'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        with open(SAVE_DIR + 'info.txt', 'w+') as f:
            f.write('MODEL = {}\nLR = {}\nBATCH_SIZE = {}\nSTRIDE = {}\nCLIP_SIZE = {}\nEPOCHS = {}'.format('I3D', LR, BATCH_SIZE, STRIDE, CLIP_SIZE, NUM_EPOCHS))
        
        train(model_num=MODEL_NUM, init_lr=LR, batch_size=BATCH_SIZE, stride=STRIDE,
              clip_size=CLIP_SIZE, num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR,
              root='/vision/group/video/scratch/jester/rgb',
              train_split='/vision/group/video/scratch/jester/annotations/jester-v1-train.csv',
              val_split='/vision/group/video/scratch/jester/annotations/jester-v1-validation.csv',
              labels='/vision/group/video/scratch/jester/annotations/jester-v1-labels.csv')

