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


class LyftDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 data_augmentation=False,
                 pre_process=False):
        self.npoints = npoints
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
                if idx % 500 == 0:
                    print("Finish %.4f%%" % (idx/len(self.datafiles)*100))
                data = np.load(file)
                npts = list(Counter(data[4]).keys())
                freq = list(Counter(data[4]).values())
                valid_obj = np.array(npts)[np.array(freq)>npoints]
                for obj in valid_obj:
                    data_select = (data.T[data[4] == obj]).T
                    new_filename = os.path.join(self.root, "objset", file[:-4].split('_')[1])+'_'+str(int(obj))
                    np.save(new_filename, data_select)
                    self.datapath.append((data_select[3,0], new_filename + ".npy")) # class, and filename
                idx += 1
            np.save(self.root + '/datapath', self.datapath)
        else:
            self.datapath = np.load(self.root + '/datapath.npy')

    def __getitem__(self, index):
        fn = self.datapath[index]
        alldata = np.load(fn[1])
        cls = fn[0]
        point_set = alldata[0:3, :]
        seg = alldata[3]

        choice = np.random.choice(seg.size, self.npoints, replace=True)
        # resample
        point_set = np.transpose(point_set[:, choice])
        cls = np.transpose(cls)
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
        cls = int(float(cls))
        cls = torch.from_numpy(np.array(cls).astype(np.int64))

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


class LyftDataset_Seg(data.Dataset):
    def __init__(self,
                 root,
                 npoints=20000,
                 data_augmentation=False,
                 pre_process=False):
        self.npoints = npoints
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
                if idx % 500 == 0:
                    print("Finish %.4f%%" % (idx/len(self.datafiles)*100))
                data = np.load(file)
                npts = list(Counter(data[4]).keys())
                freq = list(Counter(data[4]).values())
                valid_obj = np.array(npts)[np.array(freq)>npoints]
                for obj in valid_obj:
                    data_select = (data.T[data[4] == obj]).T
                    new_filename = os.path.join(self.root, "objset", file[:-4].split('_')[1])+'_'+str(int(obj))
                    np.save(new_filename, data_select)
                    self.datapath.append((data_select[3,0], new_filename + ".npy")) # class, and filename
                idx += 1
            np.save(self.root + '/datapath', self.datapath)
        else:
            self.datapath = np.load(self.root + '/datapath.npy')

    def __getitem__(self, index):
        fn = self.datapath[index]
        alldata = np.load(fn[1])
        cls = fn[0]
        point_set = alldata[0:3, :]
        seg = alldata[3]

        choice = np.random.choice(seg.size, self.npoints, replace=True)
        # resample
        point_set = np.transpose(point_set[:, choice])
        cls = np.transpose(cls)
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
        cls = int(float(cls))
        cls = torch.from_numpy(np.array(cls).astype(np.int64))

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)

