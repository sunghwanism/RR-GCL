import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c:    (N, n_h)
        # h_pl: (N, n_h)
        # h_mi: (N, n_h)
        
        sc_1 = self.f_k(h_pl, c) # (N, 1)
        sc_2 = self.f_k(h_mi, c) # (N, 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # Concatenate on dim 0 to get (2*N, 1)
        logits = torch.cat((sc_1, sc_2), 0)

        return logits
