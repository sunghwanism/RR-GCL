import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_ft, out_ft_list, activation, drop_prob, n_cls):
        super(GNNClassifier, self).__init__()

        self.proj = nn.Linear(in_ft, out_ft_list[0])

        self.conv1 = SAGEConv(out_ft_list[0], out_ft_list[1])
        self.conv2 = SAGEConv(out_ft_list[1], out_ft_list[2])
        self.conv3 = SAGEConv(out_ft_list[2], out_ft_list[3])

        self.classifier = nn.Linear(out_ft_list[3], n_cls)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x, edge_index):
        x = self.act(self.proj(x))
        x = self.drop(x)
        
        # 1. Layer 1
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        
        # 2. Layer 2
        x = self.act(self.conv2(x, edge_index))
        x = self.drop(x)
        
        # 3. Layer 3
        x = self.act(self.conv3(x, edge_index))
        x = self.drop(x)
        
        # 4. Classifier
        x = self.classifier(x)

        return x