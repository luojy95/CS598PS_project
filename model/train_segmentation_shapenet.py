# --------------------------------------------------------
# Segmentation Training implementation (for ShapeNet)
# Licensed under The MIT License [see LICENSE for details]
# Author: Jiayi Luo, Modified from Qi, 2017
# --------------------------------------------------------
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
from pointnet.Segmentation_Layer import Segmentation_Layer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
# opt.manualSeed = 100
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=None)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=None,
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'
testacc = []
trainacc = []
# classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
classifier = Segmentation_Layer()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

loss_r = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for m, data in enumerate(dataloader, 0):
        points, target = data
        gt_pts = points.transpose(2, 1)
        batch_size = points.shape[0]
        gt_pts, gt_labels = gt_pts.cuda(), target.cuda() - 1
        gt_pts = gt_pts.float()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred_labels_dist, pred_labels, pred_numobj_dist, pred_numobj, trans = classifier(gt_pts)

        # Calculate the num_obj loss
        gt_numobj,_ = torch.max(gt_labels, dim=1)
        # cls_loss = torch.sum((target_idx - pre_idx)**2)

        num_obj_loss = loss_r(pred_numobj_dist, gt_numobj)

        # Calculate the segmentation loss
        seg_loss = 0
        for b in range(batch_size):
            pred_labels_dist_ = pred_labels_dist[b].T
            pred_numobj_ = pred_numobj[b]
            pred_labels_dist_filtered = pred_labels_dist_[:,:pred_numobj_]
            # pred_labels_dist_filtered = pred_labels_dist_filtered.cuda()
            pred_labels_ = pred_labels[b]

            gt_labels_ = gt_labels[b].T
            gt_numobj_ = gt_numobj[b]

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
         
        loss = feature_transform_regularizer(trans) * 0.001 + seg_loss/2500 + num_obj_loss
        loss.backward()
        optimizer.step()
        # correct = pred_labels.eq(gt_labels.data).cpu().sum()
        correct = 0
        # total = 0
        pred_labels_cpu = pred_labels.cpu()
        gt_labels_cpu = gt_labels.cpu()
        gt_numobj_cpu,_ = torch.max(gt_labels_cpu, dim=1)
        pred_numobj_cpu = pred_numobj.cpu()
        for b in range(batch_size):
            pred_numobj_ = pred_numobj_cpu[b]
            pred_labels_ = pred_labels_cpu[b]
            gt_labels_ = gt_labels_cpu[b].T
            gt_numobj_ = gt_numobj_cpu[b]
            # print("1:",  pred_labels_.min(), pred_labels_.max())
            # print("0:", gt_labels_.min(), gt_labels_.max())
            for class_id in range(gt_numobj_ + 1):
                filter_ = gt_labels_ == class_id
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]
                if pred_labels_cls.shape[0] > 0:
                    majority_pre_id = torch.argmax(torch.Tensor([torch.sum(pred_labels_cls == i) for i in range(torch.max(pred_labels_cls).int() + 1)]))
                    match_ = majority_pre_id == pred_labels_cls
                    correct += torch.sum(match_)
                    # total += match_.shape[0]
            for class_id in range(pred_numobj_ + 1):
                filter_ = pred_labels_ == class_id
                gt_labels_cls = gt_labels_[filter_]
                pred_labels_cls = pred_labels_[filter_]
                if pred_labels_cls.shape[0] > 0:
                    majority_pre_id = torch.argmax(torch.Tensor([torch.sum(gt_labels_cls == i) for i in range(torch.max(gt_labels_cls).int() + 1)]))
                    match_ = majority_pre_id == gt_labels_cls
                    correct += torch.sum(match_)

        # print("correct:", correct/2)
        print('[%d: %d/%d] train loss: %f, seg loss: %f, numobj loss: %f, accuracy: %f' % (epoch, m, num_batch, loss.item(), seg_loss.item(), num_obj_loss.item(), correct/float(opt.batchSize * 5000)))
        trainacc.append(correct/float(opt.batchSize * 5000))
        np.save('../sample_data/trainacc',trainacc)
        if m % 1 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            gt_pts = points.transpose(2, 1)
            batch_size = points.shape[0]
            gt_pts, gt_labels = gt_pts.cuda(), target.cuda() - 1
            # print(torch.min(gt_labels))
            gt_pts = gt_pts.float()
            classifier = classifier.eval()
            pred_labels_dist, pred_labels, pred_numobj_dist, pred_numobj, _ = classifier(gt_pts)

            # Calculate the num_obj loss
            gt_numobj,_ = torch.max(gt_labels, dim=1)
            # cls_loss = torch.sum((target_idx - pre_idx)**2)

            # num_obj_loss = loss_r(pred_numobj_dist, gt_numobj)

            # Calculate the segmentation loss
            # seg_loss = 0
            # for b in range(batch_size):
            #     pred_labels_dist_ = pred_labels_dist[b].T
            #     pred_numobj_ = pred_numobj[b]
            #     pred_labels_dist_filtered = pred_labels_dist_[:,:pred_numobj_]
            #     # pred_labels_dist_filtered = pred_labels_dist_filtered.cuda()
            #     pred_labels_ = pred_labels[b]

            #     gt_labels_ = gt_labels[b].T
            #     gt_numobj_ = gt_numobj[b]
            #     # print("test %d batches: gt_obj_num=%d, pred_obj_num=%d"%(m, gt_numobj_, pred_numobj_))
            #     # GT-wise loss
            #     for class_id in range(gt_numobj_ + 1):
            #         filter_ = gt_labels_ == class_id
            #         pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
            #         gt_labels_cls = gt_labels_[filter_]
            #         pred_labels_cls = pred_labels_[filter_]
            #         if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:
            #             majority_pre_id = torch.argmax(torch.Tensor([torch.sum(pred_labels_cls == i) for i in range(torch.max(pred_labels_cls).int() + 1)]))
            #             match_ = majority_pre_id.cuda() == pred_labels_cls
            #             sum_ = torch.sum(pred_labels_dist_cls, dim=1)
            #             comp_gt_label = gt_labels_cls.new_zeros((gt_labels_cls.shape[0]))
            #             comp_pre_label_dist = pred_labels_dist_cls.new_zeros((pred_labels_dist_cls.shape[0], 2))

            #             for i in range(pred_labels_dist_cls.shape[0]):
            #                 if match_[i]:
            #                     comp_pre_label_dist[i, 0] = pred_labels_dist_cls[i, majority_pre_id]
            #                     comp_pre_label_dist[i, 1] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
            #                 elif class_id < pred_labels_dist_cls.shape[1]:
            #                     comp_pre_label_dist[i, 1] = pred_labels_dist_cls[i, majority_pre_id]
            #                     comp_pre_label_dist[i, 0] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
            #                 else:
            #                     comp_pre_label_dist[i, 0] = 0
            #                     comp_pre_label_dist[i, 1] = sum_[i]

            #             seg_loss += loss_r(comp_pre_label_dist, comp_gt_label)
                
            #     # PRED-wise loss
            #     for class_id in range(pred_numobj_ + 1):
            #         filter_ = pred_labels_ == class_id
            #         pred_labels_dist_cls = pred_labels_dist_filtered[filter_]
            #         gt_labels_cls = gt_labels_[filter_]
            #         pred_labels_cls = pred_labels_[filter_]
            #         if pred_labels_dist_cls.shape[1]*pred_labels_dist_cls.shape[0] > 0:
            #             majority_gt_id = torch.argmax(torch.Tensor([torch.sum(gt_labels_cls == i) for i in range(torch.max(gt_labels_cls).int() + 1)]))
            #             match_ = majority_pre_id.cuda() == gt_labels_cls
            #             sum_ = torch.sum(pred_labels_dist_cls, dim=1)

            #             comp_gt_label = gt_labels_cls.new_zeros((gt_labels_cls.shape[0]))
            #             comp_pre_label_dist = pred_labels_dist_cls.new_zeros((pred_labels_dist_cls.shape[0], 2))

            #             for i in range(pred_labels_dist_cls.shape[0]):
            #                 if match_[i]:
            #                     comp_pre_label_dist[i, 0] = pred_labels_dist_cls[i, majority_pre_id]
            #                     comp_pre_label_dist[i, 1] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
            #                 elif class_id < pred_labels_dist_cls.shape[1]:
            #                     comp_pre_label_dist[i, 1] = pred_labels_dist_cls[i, majority_pre_id]
            #                     comp_pre_label_dist[i, 0] = sum_[i] - pred_labels_dist_cls[i, majority_pre_id]
            #                 else:
            #                     comp_pre_label_dist[i, 0] = 0
            #                     comp_pre_label_dist[i, 1] = sum_[i]

            #             seg_loss += loss_r(comp_pre_label_dist, comp_gt_label)
            
            # loss = seg_loss + num_obj_loss
            # optimizer.step()
            # correct = pred_labels.eq(gt_labels.data).cpu().sum()
            correct = 0
            # total = 0
            pred_labels_cpu = pred_labels.cpu()
            gt_labels_cpu = gt_labels.cpu()
            gt_numobj_cpu,_ = torch.max(gt_labels_cpu, dim=1)
            pred_numobj_cpu = pred_numobj.cpu()
            for b in range(batch_size):
                pred_numobj_ = pred_numobj_cpu[b]
                pred_labels_ = pred_labels_cpu[b]
                gt_labels_ = gt_labels_cpu[b].T
                gt_numobj_ = gt_numobj_cpu[b]
                # print("1:",  pred_labels_.min(), pred_labels_.max())
                # print("0:", gt_labels_.min(), gt_labels_.max())
                for class_id in range(gt_numobj_ + 1):
                    filter_ = gt_labels_ == class_id
                    gt_labels_cls = gt_labels_[filter_]
                    pred_labels_cls = pred_labels_[filter_]
                    if pred_labels_cls.shape[0] > 0:
                        majority_pre_id = torch.argmax(torch.Tensor([torch.sum(pred_labels_cls == i) for i in range(torch.max(pred_labels_cls).int() + 1)]))
                        match_ = majority_pre_id == pred_labels_cls
                        correct += torch.sum(match_)
                        # total += match_.shape[0]
                for class_id in range(pred_numobj_ + 1):
                    filter_ = pred_labels_ == class_id
                    gt_labels_cls = gt_labels_[filter_]
                    pred_labels_cls = pred_labels_[filter_]
                    if pred_labels_cls.shape[0] > 0:
                        majority_pre_id = torch.argmax(torch.Tensor([torch.sum(gt_labels_cls == i) for i in range(torch.max(gt_labels_cls).int() + 1)]))
                        match_ = majority_pre_id == gt_labels_cls
                        correct += torch.sum(match_)

            # print("correct:", correct/2)
            print('[%d: %d/%d] test accuracy: %f' % (epoch, m, num_batch, correct/float(opt.batchSize * 5000)))
            testacc.append(correct/float(opt.batchSize * 5000))
            np.save('../sample_data/testacc',testacc)
    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

'''
## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
'''