import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Caser(nn.Module):
    def __init__(self, args=None):
        super(Caser, self).__init__()
        self.args = args
        self.seqLen = self.args.seqLen
        self.nBatch = self.args.nBatch
        self.d_model = self.args.d_model
        self.actionNum = self.args.actionNum
        self.nh = self.args.nh
        self.nv = self.args.nv
        self.activator = nn.ReLU()
        self.actionEmb = nn.Embedding(self.actionNum, self.d_model)
        self.dropout = nn.Dropout(self.args.drop)
        # vertical conv layer
        self.convV = nn.Conv2d(1, self.nv, (self.seqLen, 1))
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.seqLen - 1)]
        self.convH = nn.ModuleList([nn.Conv2d(1, self.nh, (i, self.d_model)) for i in lengths])
        self.dim_v = self.nv * self.d_model
        self.dim_h = self.nh * len(lengths)
        self.linearDim = self.dim_h + self.dim_v
        # Linear Layer
        self.T = nn.Linear(self.linearDim, self.actionNum)

    def forward(self, X):
        # X: [batch, seqLen]
        X = self.actionEmb(X).unsqueeze(1)  # [batch, 1, seqLen, d_model]
        out_v = self.convV(X)
        out_v = out_v.view(X.shape[0], -1)
        out_hs = []
        for conv in self.convH:
            convOut = self.activator(conv(X).squeeze(3))
            pool_out = F.max_pool1d(convOut, convOut.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)
        out = torch.cat([out_v, out_h], 1)
        out = self.dropout(out)
        return self.T(out)