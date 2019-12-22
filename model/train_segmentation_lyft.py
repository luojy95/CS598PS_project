# --------------------------------------------------------
# Segmentation Training implementation for LyftDataset
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
from torch.autograd import Variable

# Required parameters
epochs = 12
batch_size = 4

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

def get_match_id(pred_labels_cls, idx_list):
    if len(idx_list)>=5:
        return torch.Tensor([-1])[0].long().cuda(), idx_list
    out = []
    for i in range(torch.max(pred_labels_cls).int() + 1):
        temp = torch.sum(pred_labels_cls == i)
        out.append(temp)
    out = torch.Tensor(out)
    proposed_idx = torch.argmax(out)
    while proposed_idx in idx_list:
        if torch.max(out) == 0:
            return torch.Tensor([-1])[0].long().cuda(), idx_list
        out[proposed_idx] = 0
        proposed_idx = torch.argmax(out)
    idx_list.append(proposed_idx)
    return proposed_idx, idx_list

# Verbose
print(len(dataset), len(testdataset))

classifier = Segmentation_Layer()

# Output
trainacc = []
testacc = []
train_loss = []
train_seg_loss = []
train_geo_loss = []
test_loss = []
test_seg_loss = []
test_numobj_loss = []
test_geo_loss = []

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
        
        # Ensure to add the requires_grad = True
        pred_labels = Variable(pred_labels, requires_grad = True)
        
        # Output some of the results during the training process
        print(str(m)+"th epoch:")
        for b in range(batch_size):
            print("Pred:", end =" ")
            for i in range(5):
                idx = pred_labels[b] == i
                print(torch.sum(idx).cpu().numpy(), end =" ")
            print("GT:", end =" ")
            for i in range(5):
                idx = gt_labels[b] == i
                print(torch.sum(idx).cpu().numpy(), end =" ")
            print("|Next:|", end =" ")
        print("\n======")
        pred_labels_sm = torch.softmax(pred_labels_dist, dim=1)
        pred_numobj_dist = Variable(pred_numobj_dist, requires_grad = False)
        pred_numobj = Variable(pred_numobj, requires_grad = False)
        pred_labels = pred_labels.long()

        # Calculate the num_obj loss, use cross entropy loss, dont use square 
        gt_numobj,_ = torch.max(gt_labels, dim=1)
        num_obj_loss = loss_r(pred_numobj_dist, gt_numobj)/10
        seg_loss = 0
        geo_loss = 0
        for b in range(batch_size):

            # Geometric loss, use probability based distance (the hard assign won't be differentiable)
            pred_labels_sm_ = pred_labels_sm[b]
            gt_pts_ = gt_pts[b]
            for i in range(5):
                prob_pts_x = pred_labels_sm_[i,:] * gt_pts_[0,:]
                prob_pts_y = pred_labels_sm_[i,:] * gt_pts_[1,:]
                prob_pts_z = pred_labels_sm_[i,:] * gt_pts_[2,:]
                prob_pts_x_ = prob_pts_x - torch.mean(prob_pts_x)
                prob_pts_y_ = prob_pts_y - torch.mean(prob_pts_y)
                prob_pts_z_ = prob_pts_z - torch.mean(prob_pts_z)
                geo_loss += torch.sum(pred_labels_sm_[i,:]*(prob_pts_x_**2 + prob_pts_y_**2 + prob_pts_z_**2))

            # Update the geo_loss and
            geo_loss /= batch_size

            # Preparing the parameter for matching loss term
            pred_labels_dist_ = pred_labels_dist[b].transpose(1,0)
            pred_numobj_ = pred_numobj[b]
            pred_labels_dist_filtered = pred_labels_dist_[:,:pred_numobj_]
            pred_labels_ = pred_labels[b]

            gt_labels_ = gt_labels[b]
            gt_numobj_ = gt_numobj[b]
            gt_pts_ = gt_pts[b]

            # GT-wise loss: Fix ground truth
            used_id_list = []
            for class_id in range(gt_numobj_ + 1):
                filter_ = gt_labels_ == class_id
                pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]

                # Boundary checking
                if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:

                    # Find the majority of the label and treat as ground truth, if previously matched, change to the second majority
                    majority_pre_id, used_id_list = get_match_id(pred_labels_cls, used_id_list)

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
            used_id_list = []
            for class_id in range(pred_numobj_ + 1):
                filter_ = pred_labels_ == class_id
                pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]
                if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:

                    majority_pre_id, used_id_list = get_match_id(pred_labels_cls, used_id_list)
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
        seg_loss /= 2

        tran_loss = feature_transform_regularizer(trans) * 0.001
        loss = tran_loss + geo_loss + seg_loss

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

        print('[%d: %d/%d] train loss: %f, seg loss: %f, geo loss: %f, accuracy: %f' % (epoch, m, num_batch, loss.item(), seg_loss.item(), geo_loss.item(), float(correct)/float(batch_size * 5000)))
        
        # Save accuracy file
        trainacc.append(float(correct)/float(batch_size * 5000))
        train_loss.append(loss.item())
        train_seg_loss.append(seg_loss.item())
        train_numobj_loss.append(num_obj_loss.item())
        train_geo_loss.append(geo_loss.item())
        np.save('trainacc',trainacc)
        np.save('train_seg_loss', train_seg_loss)
        np.save('train_loss',train_loss)
        np.save('train_numobj_loss',train_numobj_loss)
        np.save('train_geo_loss',train_geo_loss)

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

            print('[%d: %d/%d] test accuracy: %f' % (epoch, m, num_batch, float(correct)/float(batch_size * 5000)))
            testacc.append(float(correct)/float(batch_size * 5000))
            np.save('testacc',testacc)
            if m%10 == 0 or m == num_batch or m == num_batch-1:
                torch.save(classifier, 'models/seg_model_trial7_'+str(epoch)+'_' + str(m))
