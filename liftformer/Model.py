import math
import torch.nn as nn
from .Layers import PositionalEncoding, Encoder


class Liftformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 receptive_field=27,
                 n_layers=6,
                 n_head=8,
                 d_in=34,
                 d_out=51,
                 d_inner=2048,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 pre_LN=True):
        super(Liftformer, self).__init__()
        self.receptive_field = receptive_field
        self.embedConv = nn.Sequential(nn.Conv1d(d_in, d_model, 1), nn.ReLU())
        self.position_enc = PositionalEncoding(d_model=d_model)
        self.encoder = Encoder(n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_k,
                               d_v=d_v,
                               d_model=d_model,
                               d_inner=d_inner,
                               dropout=dropout,
                               pre_LN=pre_LN)
        self.finalConv = nn.Conv1d(d_model, d_out, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.embedConv(x)
        x = x.transpose(1, 2)

        x = self.position_enc(x)
        x = self.encoder(x)[:, math.floor(self.receptive_field / 2):math.ceil(self.receptive_field / 2), :]

        x = x.transpose(1, 2)
        x = self.finalConv(x)
        x = x.transpose(1, 2)
        return x
