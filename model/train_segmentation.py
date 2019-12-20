# --------------------------------------------------------
# Segmentation Training implementation
# Licensed under The MIT License [see LICENSE for details]
# Author: Jiayi Luo
# --------------------------------------------------------
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from Classification_Layer import feature_transform_regularizer
from Segmentation_Layer import Segmentation_Layer
from LyftDataset_Seg import LyftDataset_Seg
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Required parameters
epochs = 5
batch_size = 32

# Manual Seed
manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Import corresponding train and test dataset
dataset = LyftDataset_Seg(root="../traindata", pre_process=False)
testdataset = LyftDataset_Seg(root="../testdata", pre_process=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Verbose
print(len(dataset), len(testdataset))

classifier = Segmentation_Layer()

# Output
trainacc = []
testacc = []

# Define the loss function, optimizer and scheduler
loss_r = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / batch_size

for epoch in range(epochs):
    scheduler.step()
    for m, data in enumerate(dataloader, 0):

        # Import the data
        points, target = data
        target = target.long()
        gt_pts = points.transpose(2, 1)
        batch_size = points.shape[0]
        gt_pts, gt_labels = gt_pts.cuda(), target.cuda()
        gt_pts = gt_pts.float()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred_labels_dist, pred_labels, pred_numobj_dist, pred_numobj, trans = classifier(gt_pts)
        pred_labels = pred_labels.long()

        # Calculate the num_obj loss, use cross entropy loss, dont use square 
        gt_numobj,_ = torch.max(gt_labels, dim=1)
        num_obj_loss = loss_r(pred_numobj_dist, gt_numobj)

        # Calculate the segmentation loss
        seg_loss = 0
        for b in range(batch_size):
            pred_labels_dist_ = pred_labels_dist[b].transpose(1,0)
            pred_numobj_ = pred_numobj[b]
            pred_labels_dist_filtered = pred_labels_dist_[:,:pred_numobj_]
            pred_labels_ = pred_labels[b]

            gt_labels_ = gt_labels[b]
            gt_numobj_ = gt_numobj[b]

            # GT-wise loss: Fix ground truth
            for class_id in range(gt_numobj_ + 1):
                filter_ = gt_labels_ == class_id
                pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]

                # Boundary checking
                if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:

                    # Find the majority of the label and treat as ground truth
                    majority_pre_id = torch.argmax(torch.Tensor([torch.sum(pred_labels_cls == i) for i in range(torch.max(pred_labels_cls).int() + 1)]))
                    
                    # Update the loass
                    match_ = majority_pre_id.cuda() == pred_labels_cls
                    sum_ = torch.sum(pred_labels_dist_cls, dim=1)

                    # Modify the labels to binary set (0, 1)
                    comp_gt_label = gt_labels_cls.new_zeros((gt_labels_cls.shape[0]))
                    comp_pre_label_dist = pred_labels_dist_cls.new_zeros((pred_labels_dist_cls.shape[0], 2))

                    # Update loss under three scenarios, combine different cross-entropy into N x 2 matrix
                    for i in range(pred_labels_dist_cls.shape[0]):

                        # If match exists, do as normal
                        if match_[i]:
                            comp_pre_label_dist[i, 0] = pred_labels_dist_cls[i, majority_pre_id]
                            comp_pre_label_dist[i, 1] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                        
                        # If match doesn't exist
                        # If class_id is smaller than max possible one, just add the rest to the last slot and treat is as all wrong
                        elif class_id < pred_labels_dist_cls.shape[1]:
                            comp_pre_label_dist[i, 1] = pred_labels_dist_cls[i, majority_pre_id]
                            comp_pre_label_dist[i, 0] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                        
                        # If class_id exceeds the boundary, just make the correct slot 0
                        else:
                            comp_pre_label_dist[i, 0] = 0
                            comp_pre_label_dist[i, 1] = sum_[i]
                    seg_loss += loss_r(comp_pre_label_dist, comp_gt_label)
            
            # PRED-wise loss: Fix predicted label
            # Similar annotation as before, just skip
            for class_id in range(pred_numobj_ + 1):
                filter_ = pred_labels_ == class_id
                pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]
                if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:
                    majority_gt_id = torch.argmax(torch.Tensor([torch.sum(gt_labels_cls == i) for i in range(torch.max(gt_labels_cls).int() + 1)]))
                    match_ = majority_pre_id.cuda() == gt_labels_cls
                    sum_ = torch.sum(pred_labels_dist_cls, dim=1)

                    comp_gt_label = gt_labels_cls.new_zeros((gt_labels_cls.shape[0]))
                    comp_pre_label_dist = pred_labels_dist_cls.new_zeros((pred_labels_dist_cls.shape[0], 2))

                    for i in range(pred_labels_dist_cls.shape[0]):
                        if match_[i]:
                            comp_pre_label_dist[i, 0] = pred_labels_dist_cls[i, majority_pre_id]
                            comp_pre_label_dist[i, 1] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                        elif class_id < pred_labels_dist_cls.shape[1]:
                            comp_pre_label_dist[i, 1] = pred_labels_dist_cls[i, majority_pre_id]
                            comp_pre_label_dist[i, 0] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                        else:
                            comp_pre_label_dist[i, 0] = 0
                            comp_pre_label_dist[i, 1] = sum_[i]

                    seg_loss += loss_r(comp_pre_label_dist, comp_gt_label)
        # Update seg_lss with constant 1/2500
        loss = feature_transform_regularizer(trans) * 0.001 + seg_loss/2500 + num_obj_loss

        # Update loss
        loss.backward()
        optimizer.step()

        # Update correct
        # Similar annotation as before: Majority Matching
        correct = 0
        pred_labels_cpu = pred_labels.cpu()
        gt_labels_cpu = gt_labels.cpu()
        gt_numobj_cpu,_ = torch.max(gt_labels_cpu, dim=1)
        pred_numobj_cpu = pred_numobj.cpu()
        for b in range(batch_size):
            pred_numobj_ = pred_numobj_cpu[b]
            pred_labels_ = pred_labels_cpu[b]
            gt_labels_ = gt_labels_cpu[b]
            gt_numobj_ = gt_numobj_cpu[b]
            for class_id in range(gt_numobj_ + 1):
                filter_ = gt_labels_ == class_id
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]
                if pred_labels_cls.shape[0] > 0:
                    majority_pre_id = torch.argmax(torch.Tensor([torch.sum(pred_labels_cls == i) for i in range(torch.max(pred_labels_cls).int() + 1)]))
                    match_ = majority_pre_id == pred_labels_cls
                    correct += torch.sum(match_)
            for class_id in range(pred_numobj_ + 1):
                filter_ = pred_labels_ == class_id
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]
                if pred_labels_cls.shape[0] > 0:
                    majority_pre_id = torch.argmax(torch.Tensor([torch.sum(gt_labels_cls == i) for i in range(torch.max(gt_labels_cls).int() + 1)]))
                    match_ = majority_pre_id == gt_labels_cls
                    correct += torch.sum(match_)

        print('[%d: %d/%d] train loss: %f, seg loss: %f, numobj loss: %f, accuracy: %f' % (epoch, m, num_batch, loss.item(), seg_loss.item(), num_obj_loss.item(), float(correct)/float(batch_size * 5000)))
        
        # Save accuracy file
        trainacc.append(float(correct)/float(batch_size * 5000))
        np.save('trainacc',trainacc)

        # Test subroutine
        if m % 1 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target.long()
            gt_pts = points.transpose(2, 1)
            batch_size = points.shape[0]
            gt_pts, gt_labels = gt_pts.cuda(), target.cuda()
            # print(torch.min(gt_labels))
            gt_pts = gt_pts.float()
            classifier = classifier.eval()
            pred_labels_dist, pred_labels, pred_numobj_dist, pred_numobj, _ = classifier(gt_pts)
            pred_labels = pred_labels.long()
            # Calculate the num_obj loss
            gt_numobj,_ = torch.max(gt_labels, dim=1)


            ''' 
            # Uncomment is want to calculate the test loss!!!

            cls_loss = torch.sum((target_idx - pre_idx)**2)

            num_obj_loss = loss_r(pred_numobj_dist, gt_numobj)

            Calculate the segmentation loss
            seg_loss = 0
            for b in range(batch_size):
                pred_labels_dist_ = pred_labels_dist[b].T
                pred_numobj_ = pred_numobj[b]
                pred_labels_dist_filtered = pred_labels_dist_[:,:pred_numobj_]
                # pred_labels_dist_filtered = pred_labels_dist_filtered.cuda()
                pred_labels_ = pred_labels[b]

                gt_labels_ = gt_labels[b].T
                gt_numobj_ = gt_numobj[b]
                # print("test %d batches: gt_obj_num=%d, pred_obj_num=%d"%(m, gt_numobj_, pred_numobj_))
                # GT-wise loss
                for class_id in range(gt_numobj_ + 1):
                    filter_ = gt_labels_ == class_id
                    pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
                    gt_labels_cls = gt_labels_[filter_]
                    pred_labels_cls = pred_labels_[filter_]
                    if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:
                        majority_pre_id = torch.argmax(torch.Tensor([torch.sum(pred_labels_cls == i) for i in range(torch.max(pred_labels_cls).int() + 1)]))
                        match_ = majority_pre_id.cuda() == pred_labels_cls
                        sum_ = torch.sum(pred_labels_dist_cls, dim=1)
                        comp_gt_label = gt_labels_cls.new_zeros((gt_labels_cls.shape[0]))
                        comp_pre_label_dist = pred_labels_dist_cls.new_zeros((pred_labels_dist_cls.shape[0], 2))

                        for i in range(pred_labels_dist_cls.shape[0]):
                            if match_[i]:
                                comp_pre_label_dist[i, 0] = pred_labels_dist_cls[i, majority_pre_id]
                                comp_pre_label_dist[i, 1] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                            elif class_id < pred_labels_dist_cls.shape[1]:
                                comp_pre_label_dist[i, 1] = pred_labels_dist_cls[i, majority_pre_id]
                                comp_pre_label_dist[i, 0] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                            else:
                                comp_pre_label_dist[i, 0] = 0
                                comp_pre_label_dist[i, 1] = sum_[i]

                        seg_loss += loss_r(comp_pre_label_dist, comp_gt_label)
                
                # PRED-wise loss
                for class_id in range(pred_numobj_ + 1):
                    filter_ = pred_labels_ == class_id
                    pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
                    gt_labels_cls = gt_labels_[filter_]
                    pred_labels_cls = pred_labels_[filter_]
                    if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:
                        majority_gt_id = torch.argmax(torch.Tensor([torch.sum(gt_labels_cls == i) for i in range(torch.max(gt_labels_cls).int() + 1)]))
                        match_ = majority_pre_id.cuda() == gt_labels_cls
                        sum_ = torch.sum(pred_labels_dist_cls, dim=1)

                        comp_gt_label = gt_labels_cls.new_zeros((gt_labels_cls.shape[0]))
                        comp_pre_label_dist = pred_labels_dist_cls.new_zeros((pred_labels_dist_cls.shape[0], 2))

                        for i in range(pred_labels_dist_cls.shape[0]):
                            if match_[i]:
                                comp_pre_label_dist[i, 0] = pred_labels_dist_cls[i, majority_pre_id]
                                comp_pre_label_dist[i, 1] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                            elif class_id < pred_labels_dist_cls.shape[1]:
                                comp_pre_label_dist[i, 1] = pred_labels_dist_cls[i, majority_pre_id]
                                comp_pre_label_dist[i, 0] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
                            else:
                                comp_pre_label_dist[i, 0] = 0
                                comp_pre_label_dist[i, 1] = sum_[i]

                        seg_loss += loss_r(comp_pre_label_dist, comp_gt_label)
            
            loss = seg_loss + num_obj_loss

            '''
            correct = 0
            pred_labels_cpu = pred_labels.cpu()
            gt_labels_cpu = gt_labels.cpu()
            gt_numobj_cpu,_ = torch.max(gt_labels_cpu, dim=1)
            pred_numobj_cpu = pred_numobj.cpu()
            for b in range(batch_size):
                pred_numobj_ = pred_numobj_cpu[b]
                pred_labels_ = pred_labels_cpu[b]
                gt_labels_ = gt_labels_cpu[b]
                gt_numobj_ = gt_numobj_cpu[b]
                for class_id in range(gt_numobj_ + 1):
                    filter_ = gt_labels_ == class_id
                    gt_labels_cls = gt_labels_[filter_]
                    pred_labels_cls = pred_labels_[filter_]
                    if pred_labels_cls.shape[0] > 0:
                        majority_pre_id = torch.argmax(torch.Tensor([torch.sum(pred_labels_cls == i) for i in range(torch.max(pred_labels_cls).int() + 1)]))
                        match_ = majority_pre_id == pred_labels_cls
                        correct += torch.sum(match_)
                for class_id in range(pred_numobj_ + 1):
                    filter_ = pred_labels_ == class_id
                    gt_labels_cls = gt_labels_[filter_]
                    pred_labels_cls = pred_labels_[filter_]
                    if pred_labels_cls.shape[0] > 0:
                        majority_pre_id = torch.argmax(torch.Tensor([torch.sum(gt_labels_cls == i) for i in range(torch.max(gt_labels_cls).int() + 1)]))
                        match_ = majority_pre_id == gt_labels_cls
                        correct += torch.sum(match_)

            print("correct:", correct/2)
            print('[%d: %d/%d] test accuracy: %f' % (epoch, m, num_batch, float(correct)/float(batch_size * 5000)))
            testacc.append(float(correct)/float(batch_size * 5000))
            np.save('testacc',testacc)
