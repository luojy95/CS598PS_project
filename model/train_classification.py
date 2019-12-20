# --------------------------------------------------------
# Classification Training implementation
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
from LyftDataset import LyftDataset
from Classification_Layer import Classification_Layer, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Blue text for output
blue = lambda x: '\033[94m' + x + '\033[0m'

batch_size = 64
dataset = LyftDataset(root="../traindata", npoints=1600, pre_process=False)
testdataset = LyftDataset(root="../testdata", npoints=1600, pre_process=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Verbose
print(len(dataset), len(testdataset))

# Define the loss function, optimizer and scheduler
classifier = Classification_Layer(num_classes=9)
criterion = nn.CrossEntropyLoss()
num_batch = len(dataset) / batch_size

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
nepoch = 10

# The global variable
test_acc = []
train_acc = []
test_loss = []
train_loss = []
for epoch in range(nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target, _ = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        points = points.float()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans = classifier(points)
        loss = criterion(pred, target)

        # Apply regularization factor
        loss += feature_transform_regularizer(trans) * 0.001

        # Update loss
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        # Recoding subroutine
        if i % 1 == 0:
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / batch_size))
            train_acc.append(correct.item() / batch_size)
            train_loss.append(loss.item())
            np.save("train_acc", train_acc)
            np.save("train_loss", train_loss)
        
        # Test subroutine
        if i % 1 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, _ = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            points = points.float()
            classifier = classifier.eval()
            pred, trans = classifier(points)
            loss = criterion(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(batch_size)))
            test_acc.append(correct.item() / float(batch_size))
            test_loss.append(loss.item())
            np.save("test_acc", test_acc)
            np.save("test_loss", test_loss)
torch.save(classifier, 'classifier.model')

# Calculate the overall test accuracy
total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target, _ = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    points = points.float()
    classifier = classifier.eval()
    pred, trans = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
np.save("final_acc", total_correct / float(total_testset))