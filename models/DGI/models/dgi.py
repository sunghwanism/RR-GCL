import torch
import torch.nn as nn
from ..layers import DenseGCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in_numeric, n_uniprot, n_bin, emb_dim_uniprot, emb_dim_bin, out_dim_list, activation, drop_prob):
        super(DGI, self).__init__()
        
        self.uniprot_embedding = nn.Embedding(n_uniprot, emb_dim_uniprot)
        self.bin_embedding = nn.Embedding(n_bin, emb_dim_bin)

        total_in_channels = n_in_numeric + emb_dim_uniprot + emb_dim_bin
        
        self.gcn = DenseGCN(total_in_channels, out_dim_list, activation, drop_prob)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim_list[-1])

    def _get_combined_feat(self, x_numeric, uniprot_idx, bin_idx):
        u_emb = self.uniprot_embedding(uniprot_idx) # (N, emb_dim)
        b_emb = self.bin_embedding(bin_idx)       # (N, emb_dim)
        
        return torch.cat([x_numeric, u_emb, b_emb], dim=1)

    def forward(self, x_num, uniprot_idx, bin_idx, 
                shuf_num, shuf_uniprot, shuf_bin, 
                edge_index, batch=None, samp_bias1=None, samp_bias2=None):
        
        x = self._get_combined_feat(x_num, uniprot_idx, bin_idx)
        shuf_x = self._get_combined_feat(shuf_num, shuf_uniprot, shuf_bin)

        h_1 = self.gcn(x, edge_index)
        c = self.read(h_1, batch)
        c = self.sigm(c)

        h_2 = self.gcn(shuf_x, edge_index)

        if batch is None:
            c_expanded = c.expand_as(h_1)
        else:
            c_expanded = c[batch]

        ret = self.disc(c_expanded, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    def embed(self, x_num, uniprot_idx, bin_idx, edge_index, batch=None):
        
        x = self._get_combined_feat(x_num, uniprot_idx, bin_idx)
        h_1 = self.gcn(x, edge_index)
        c = self.read(h_1, batch)
        return h_1.detach(), c.detach()