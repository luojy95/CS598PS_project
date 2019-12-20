# --------------------------------------------------------
# Segmentation Layer implementation
# Licensed under The MIT License [see LICENSE for details]
# Author: Jiayi Luo
# --------------------------------------------------------

import torch 
import torch.nn as nn
import torch.nn.functional as F
from Feat_Map import Feat_Map

class Segmentation_Layer(nn.Module):
    '''
        Input shape: [batchsize, 3, numberofpts]
    '''
    def __init__(self, reception_ratio=0.05, num_pts=2500):
        super(Segmentation_Layer, self).__init__()
        field = int(num_pts*reception_ratio)
        self.dilation = max(field//15, 1)
        self.num_pts = num_pts

        # Preload the Feat_Map module
        self.TOP = Feat_Map(reception_ratio=reception_ratio, num_pts=1000)
        
        # Define the num_obj branch prediction conv layers
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 128, 1)
        self.fc1 = nn.Linear(self.num_pts*128, 40)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Define the object label prediction conv layers
        self.conv3 = nn.Conv1d(1088, 512, 1)
        self.conv4 = nn.Conv1d(512, 256, 1)
        self.conv5 = nn.Conv1d(256, 40, 1)
        # self.fc2 = nn.Linear(self.num_pts*32, self.num_pts*40)

        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_keep, x, trans = self.TOP(x)
        
        # Concat to global feature map and the feature
        x_ = x.unsqueeze(2)
        x_ = x_.repeat(1, 1, x_keep.shape[2])
        feat = torch.cat((x_keep, x_), dim=1)

        # The num_obj prediction branch
        x = F.relu(self.bn1(self.conv1(feat)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        pred_numobj_dist = x.view(batch_size, -1)

        pred_numobj = torch.argmax(pred_numobj_dist, dim = 1) + 1

        # Based on the predicted idx, obtain the prediction for each point
        x = F.relu(self.bn3(self.conv3(feat)))
        x = F.relu(self.bn4(self.conv4(x)))
        pred_label_dist = self.conv5(x)

        # For each scene in the batch, get it's corresponding trimmed ground truch label
        pred_label = x.new_zeros((batch_size, self.num_pts))
        for b in range(batch_size):
            x_ = pred_label_dist[b][:pred_numobj[b]]
            label = torch.argmax(x_, dim=0)
            pred_label[b] = label
        return pred_label_dist, pred_label, pred_numobj_dist, pred_numobj, trans
        
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == "__main__":
    x = torch.randn(32, 3, 2500)
    a = Segmentation_Layer()
    x,y,z,m,_ = a(x)  
    print(x.shape)
    print(y)
    print(z.shape)
    print(m)