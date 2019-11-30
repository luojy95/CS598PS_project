# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.nn.functional as F
from Feat_Map import Feat_Map

class Segmentation_Layer(nn.Module):
    '''
        Input shape: [batchsize, 3, numberofpts]
    '''
    def __init__(self, num_classes, reception_ratio=0.05, num_pts=2500):
        super(Segmentation_Layer, self).__init__()
        field = int(num_pts*reception_ratio)
        self.dilation = field//15
        
        # Preload the Feat_Map module
        self.TOP = Feat_Map(reception_ratio=reception_ratio, num_pts=num_pts)
        
        # Define the mlp layer infor
        self.conv1 = nn.Conv1d(1088, 512, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        self.conv4 = nn.Conv1d(128, num_classes, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x_keep, x = self.TOP(x)
        
        # Concat to global feature map and the feature
        x_ = x.unsqueeze(2)
        x_ = x_.repeat(1, 1, x_keep.shape[2])
        x = torch.cat((x_keep, x_), dim=1)
        
        # Apply the following convlayer mlp layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        return x.permute(0, 2, 1)

if __name__ == "__main__":
    x = torch.randn(32, 3, 2500)
    a = Segmentation_Layer(10)
    x = a(x)  
        