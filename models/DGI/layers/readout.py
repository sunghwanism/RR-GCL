import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, batch=None):
        return global_mean_pool(seq, batch)
