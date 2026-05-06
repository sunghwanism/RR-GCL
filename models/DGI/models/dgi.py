import torch
import torch.nn as nn
from models.DGI.layers import DenseGCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in_numeric, cat_feat_num_dict, emb_dim, out_dim_list, activation, drop_prob):
        super(DGI, self).__init__()
        
        self.cat_feat_emb_dict = nn.ModuleDict({
            key: nn.Embedding(value, emb_dim[key], padding_idx=0) for key, value in cat_feat_num_dict.items()
        })

        total_in_channels = n_in_numeric + sum(emb_dim[key] for key in cat_feat_num_dict.keys())
        
        self.gcn = DenseGCN(total_in_channels, out_dim_list, activation, drop_prob)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim_list[-1])

    def _get_combined_feat(self, x_numeric, cat_feat_dict):
        cat_feat_emb_list = []
        for key in sorted(self.cat_feat_emb_dict.keys()):
            if key in cat_feat_dict:
                feat = cat_feat_dict[key]
                emb = self.cat_feat_emb_dict[key](feat)
                if emb.dim() == 3:
                    emb = emb.sum(dim=1)
                cat_feat_emb_list.append(emb)
        
        if cat_feat_emb_list:
            return torch.cat([x_numeric] + cat_feat_emb_list, dim=1)
        return x_numeric

    def forward(self, x_num, cat_feats, 
                shuf_num, shuf_cat_feats, 
                edge_index, batch=None, samp_bias1=None, samp_bias2=None):
        
        x = self._get_combined_feat(x_num, cat_feats)
        shuf_x = self._get_combined_feat(shuf_num, shuf_cat_feats)

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

    def embed(self, x_num, cat_feats, edge_index, batch=None):
        
        x = self._get_combined_feat(x_num, cat_feats)
        h_1 = self.gcn(x, edge_index)
        c = self.read(h_1, batch)
        return h_1.detach(), c.detach()