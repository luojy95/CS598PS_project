import torch 
import torch.nn as nn

class T_Net(nn.Module):
    '''
        Input shape: [batchsize, 3, numberofpts]
    '''
    def __init__(self, in_dim=3, out_dim=3, num_pts=2500, num_blocks=3, reception_ratio=0.05, maxDim = 1024, minDim = 64):
        super(T_Net, self).__init__()
        self.maxDim = maxDim
        self.minDim = minDim
        
        self.out_dim = out_dim
        self.num_pts = num_pts
        self.num_blocks = num_blocks
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(minDim, out_dim**2)
        
        # Calculate the reception area
        field = int(num_pts*reception_ratio)
        self.dilation = field//(num_blocks*3)

        # Calculate the block parameters
        self.linear_dim = [maxDim, 512, 256, minDim] if minDim <= 128 else [maxDim, 512, minDim]
        self.layer_dim = [in_dim, minDim]
        for i in range(num_blocks - 1):
            self.layer_dim.append(self.layer_dim[-1]*4) if self.layer_dim[-1]*4 <= maxDim else self.layer_dim.append(maxDim)
        self.layer_dim.append(maxDim)
        
        # Construct the block
        self.convs = self.get_convs()
        self.linears = self.get_linears()

    def block(self, in_dim, out_dim):
        return nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        
    def get_convs(self):
        layers = []
        for i in range(self.num_blocks):
            layers.append(self.block(self.layer_dim[i], self.layer_dim[i + 1]))
            layers.append(nn.BatchNorm1d(self.layer_dim[i + 1]))
            layers.append(self.relu)
        return nn.Sequential(*layers)
    
    def get_linears(self):
        layers = []
        for i in range(len(self.linear_dim) - 1):
            layers.append(nn.Linear(self.linear_dim[i], self.linear_dim[i + 1]))
            layers.append(nn.BatchNorm1d(self.linear_dim[i + 1]))
            layers.append(self.relu)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)
        
        # Drop all the redundent terms, accelerate the convergence rate
        x = torch.max(x, 2)[0]
        x = self.linears(x)
        x = self.fc(x)
        x = x.view(-1, self.out_dim, self.out_dim)
        return x

if __name__ == "__main__":   
    net = T_Net(in_dim=64, out_dim=64, minDim = 256)
    print(net)
    #net_new = STNkd()
    a = torch.randn(32, 64, 2500)
    #c = net_new(a)
    b = net(a)
    #print(c.shape)