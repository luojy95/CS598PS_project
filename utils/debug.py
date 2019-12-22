# --------------------------------------------------------
# Debug Module: Visulization, Model Load functions
# Licensed under The MIT License [see LICENSE for details]
# Author: Jiayi Luo
# --------------------------------------------------------

import torch
import numpy as np
from LyftDataset_Seg import LyftDataset_Seg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
seed = 722
torch.manual_seed(seed)

color_classes = {
    0:'g',
    1:'r',
    2:'y',
    3:'b',
    4:'gray',
    5:'deepskyblue',
    6:'navy',
    7:'sandybrown',
    8:'firebrick',
    9:'palevioletred',
    10:'deeppink',
    11:'crimson',
    12:'b',
    13:'g',
    14:'olive',
    15:'r',
    16:'mediumpurple',
    17:'royalblue',
    18:'darkred',
    19:'springgreen',
    20:'b'  
}

# Import corresponding train and test dataset
dataset = LyftDataset_Seg(root="../traindata", pre_process=False)
testdataset = LyftDataset_Seg(root="../testdata", pre_process=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=2, shuffle=True, num_workers=8)

classifier = torch.load('models/seg_model_trial7_0_90')
classifier = classifier.cpu()
classifier.eval()

for m, data in enumerate(dataloader, 0):
    # if m <= 1000:
    #     continue
    points, target = data
    target = target.long()
    gt_pts = points.transpose(2, 1)
    # print(gt_pts.shape)

    gt_pts = gt_pts.float()
    _, pred_labels, _,_,_ = classifier(gt_pts)
    point = gt_pts.numpy()[0]
    pred_label = pred_labels.numpy()[0]
    target = target.numpy()[0]
    idx_list = []
    print(str(m) + " epoch: Predicted")
    for i in range(5):
        idx = pred_label == i
        print(i, np.sum(idx))
        idx_list.append(np.sum(idx))
    idx_list.sort()
    print("GT:")
    for i in range(5):
        idx = target == i
        print(i, np.sum(idx))

    if abs(idx_list[-1] - idx_list[-2]) < 500 and abs(idx_list[-2] - idx_list[-3]) < 200:
        print("Start Plotting") 
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(5):
            idx = pred_label == i
            idx_list.append(np.sum(idx))
            point_ = (point.T[idx]).T
            ax.scatter(point_[0,:], point_[1,:], point_[2,:], color = color_classes[i])
        plt.savefig('output/out_' + str(m) +'.png')
        break
