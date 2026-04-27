import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from utils.activation import getActivation

class DenseGCN(nn.Module):
    def __init__(self, in_ft, out_ft_list, activation, drop_prob):
        super(DenseGCN, self).__init__()
        self.proj = nn.Linear(in_ft, out_ft_list[0])
        self.conv1 = SAGEConv(in_ft, out_ft_list[0], aggr='mean')
        self.conv2 = SAGEConv(out_ft_list[0] + out_ft_list[0], out_ft_list[1], aggr='mean')
        self.conv3 = SAGEConv(out_ft_list[0] + out_ft_list[0] + out_ft_list[1], out_ft_list[2], aggr='mean')

        self.act = getActivation(activation)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x, edge_index):
        x_p = self.act(self.proj(x)) # (N, out_ft_list[0])
        x_p = self.drop(x_p)
        
        # Layer 1
        h1 = self.act(self.conv1(x, edge_index))# (N, out_ft_list[0])
        h1 = self.drop(h1)
        
        # Layer 2
        in2 = torch.cat([x_p, h1], dim=1) # (N, out_ft_list[0] + out_ft_list[0])
        h2 = self.act(self.conv2(in2, edge_index))
        h2 = self.drop(h2)
        
        # Layer 3
        in3 = torch.cat([x_p, h1, h2], dim=1) # (N, out_ft_list[0] + out_ft_list[0] + out_ft_list[1])
        h3 = self.act(self.conv3(in3, edge_index))
        h3 = self.drop(h3)
        
        return h3
