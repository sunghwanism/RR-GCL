import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from utils.activation import getActivation

class GATClassifier(torch.nn.Module):
    def __init__(self, in_ft, out_ft_list, activation, drop_prob, n_cls, heads=4):
        super(GATClassifier, self).__init__()

        self.proj = nn.Linear(in_ft, out_ft_list[0])

        self.conv1 = GATConv(out_ft_list[0], out_ft_list[1], heads=heads, concat=True)
        self.conv2 = GATConv(out_ft_list[1] * heads, out_ft_list[2], heads=heads, concat=True)
        self.conv3 = GATConv(out_ft_list[2] * heads, out_ft_list[3], heads=1, concat=False)

        self.classifier = nn.Linear(out_ft_list[3], n_cls)

        self.act = getActivation(activation)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x, edge_index):
        x = self.act(self.proj(x))
        x = self.drop(x)
        
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        
        x = self.act(self.conv2(x, edge_index))
        x = self.drop(x)
        
        x = self.act(self.conv3(x, edge_index))
        x = self.drop(x)
        
        x = self.classifier(x)

        return x