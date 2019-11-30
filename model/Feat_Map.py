# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
from T_Net import T_Net
import torch.nn.functional as F

class Feat_Map(nn.Module):
    '''
        Input shape: [batchsize, 3, numberofpts]
    '''
    def __init__(self, reception_ratio=0.05, num_pts=2500):
        super(Feat_Map, self).__init__()
        # Calculate the reception area
        field = int(num_pts*reception_ratio)
        self.dilation = field//15
        
        self.input_trans = T_Net(in_dim=3, out_dim=3, minDim=64)
        self.feat_trans = T_Net(in_dim=64, out_dim=64, minDim=256)
        
        # Define each convlayer
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        self.conv3 = nn.Conv1d(256, 1024, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        
        # BatchNorm layer
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        self.num_pts = x.shape[2]
        
        # Perform matrix multiply
        x_ = x.permute(0, 2, 1)
        x_ = torch.bmm(x_, self.input_trans(x))
        x = x_.permute(0, 2, 1)
        
        # First mlp
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Perform matrix multiply
        x_ = x.permute(0, 2, 1)
        x_ = torch.bmm(x_, self.feat_trans(x))
        x = x_.permute(0, 2, 1)
        
        # Clone this feature map for future usage
        x_keep = x.clone()
        
        # Second mlp
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Maxpooling from n x 1024 ==> 1 x 1024
        x = torch.max(x, 2)[0]
        
        return x_keep, x
        
if __name__ == "__main__":
    x = torch.randn(32, 3, 2500)
    a = Feat_Map()
    x_keep, x = a(x)
    x_ = x.unsqueeze(2)
    x_ = x_.repeat(1, 1, x_keep.shape[2])
    x = torch.cat((x_keep, x_), dim=1)