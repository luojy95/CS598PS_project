# --------------------------------------------------------
# LyftDataset Dataset
# Licensed under The MIT License [see LICENSE for details]
# Author: Zixu Zhao, Jiayi Luo
# --------------------------------------------------------

from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import json
import glob
from collections import Counter


class LyftDataset_Seg(data.Dataset):
    def __init__(self,
                 root,
                 n_total_points=2500,
                 n_obj_points=500,
                 data_augmentation=False,
                 pre_process=False):
        self.npoints = n_total_points
        self.root = root
        self.catfile = os.path.join(self.root, 'class.txt')
        self.cat = []
        self.data_augmentation = data_augmentation

        with open(self.catfile, 'r') as f:
            for line in f:
                self.cat.append(line)

        self.meta = {}
        self.datafiles = glob.glob(os.path.join(self.root, "dataset", "*.npy"))

        for item in self.cat:
            self.meta[item] = []

        if pre_process:
            self.datapath = []
            idx = 0
            for file in self.datafiles:
                if idx % 10 == 0:
                    print("Finish %.4f%%" % (idx/len(self.datafiles)*100))
                data = np.load(file)
                npts = list(Counter(data[4]).keys())
                freq = list(Counter(data[4]).values())
                valid_obj = np.array(npts)[np.array(freq) > n_obj_points]
                if valid_obj.size < 1:
                    continue

                if valid_obj.size > 5:
                    freq = np.array(freq)[np.array(freq) > n_obj_points]
                    sort_index = np.argsort(freq)
                    valid_obj = valid_obj[np.flip(sort_index)]
                    valid_obj = valid_obj[0:5]

                data_select = (data.T[np.isin(data[4], valid_obj)]).T

                obj0 = data_select[4,] == valid_obj[0]
                if valid_obj.size >=2:
                    obj1 = data_select[4,] == valid_obj[1]
                if valid_obj.size >= 3:
                    obj2 = data_select[4,] == valid_obj[2]
                if valid_obj.size >= 4:
                    obj3 = data_select[4,] == valid_obj[3]
                if valid_obj.size >= 5:
                    obj4 = data_select[4,] == valid_obj[4]

                data_select[4, obj0] = valid_obj.tolist().index(valid_obj[0])
                if valid_obj.size >= 2:
                    data_select[4, obj1] = valid_obj.tolist().index(valid_obj[1])
                if valid_obj.size >= 3:
                    data_select[4, obj2] = valid_obj.tolist().index(valid_obj[2])
                if valid_obj.size >= 4:
                    data_select[4, obj3] = valid_obj.tolist().index(valid_obj[3])
                if valid_obj.size >= 5:
                    data_select[4, obj4] = valid_obj.tolist().index(valid_obj[4])

                new_filename = os.path.join(self.root, "sceneset", file[:-4].split('_')[1])
                np.save(new_filename, data_select)
                self.datapath.append((idx, new_filename + ".npy"))
                idx += 1
            np.save(self.root + '/datapath', self.datapath)
        else:
            self.datapath = np.load(self.root + '/datapath.npy')

    def __getitem__(self, index):
        fn = self.datapath[index]
        alldata = np.load(fn[1])
        point_set = alldata[0:3, :] ## 3 by n
        seg = alldata[4] ## 1 by n

        choice = np.random.choice(seg.size, self.npoints, replace=True)
        # resample
        point_set = np.transpose(point_set[:, choice])
        seg = np.transpose(seg)

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)

        return point_set, seg

    def __len__(self):
        return len(self.datapath)