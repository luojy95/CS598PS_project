# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.nn.functional as F
from pointnet.Feat_Map import Feat_Map

class Classification_Layer(nn.Module):
    '''
        Input shape: [batchsize, 3, numberofpts]
    '''
    def __init__(self, num_classes, reception_ratio=0.05, num_pts=2500):
        super(Classification_Layer, self).__init__()
        
        # Preload the Feat_Map module
        self.TOP = Feat_Map(reception_ratio=reception_ratio, num_pts=num_pts)
        
        # Define the rest fully connected layer
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Define dropout layer to avoid overfitting
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        _, x, trans = self.TOP(x)
        x = F.relu(self.bn1(self.dropout(self.fc1(x))))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans

if __name__ == "__main__":
    x = torch.randn(32, 3, 2500)
    a = Classification_Layer(10)
    x = a(x)      
        