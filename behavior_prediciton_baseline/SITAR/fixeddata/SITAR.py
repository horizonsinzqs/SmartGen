import torch
import torch.nn as nn
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class STAR(nn.Module):
    def __init__(self, sequenceLength=10, d_model=40, timeStamp=7 * 8, actionNum=268): # d_model as recommended in the paper
        super(STAR, self).__init__()
        self.E = clones(nn.Linear(1, d_model).to(device), sequenceLength-1)
        self.Q = clones(nn.Linear(d_model, d_model).to(device), sequenceLength-1)
        self.L = clones(nn.Linear(d_model, d_model * d_model).to(device), sequenceLength-1)
        self.M = clones(nn.Linear(d_model, d_model).to(device), sequenceLength-1)
        self.T = nn.Linear(d_model, actionNum).to(device)
        self.d = d_model
        self.sl = sequenceLength
        self.actionNum = actionNum
        self.actionEmb = nn.Embedding(actionNum, d_model).to(device)
        self.act = nn.ReLU6()


    def forward(self, X, t1, t2):
        p0 = torch.zeros(X.shape[0], self.d)
        h0 = torch.zeros(X.shape[0], self.d)
        ActionEmb = self.actionEmb(X[:,:,3])
        h0, p0, ActionEmb = h0.to(device), p0.to(device), ActionEmb.to(device)
        t = t1 * 8 + t2
        t = t.to(device)
        for i in range(self.sl - 1):
            p1 = self.act(self.E[i](torch.unsqueeze(t[:,i], 1).float().to(device)) + self.Q[i](p0)).to(device)
            # p1 = self.E[i](torch.unsqueeze(t[:, i], 1).float().to(device)) + self.Q[i](p0)
            jt = self.act(self.L[i](p1)).to(device)
            # jt = self.L[i](p1)
            jt = torch.reshape(jt, (jt.shape[0], self.d, self.d))
            # tList = []
            h00 = torch.unsqueeze(h0, dim=1)
            tt = torch.squeeze(torch.matmul(h00, jt))
            # for j in range(X.shape[0]):
            #     tList.append(torch.matmul(h0[j], jt[j]))
            # tt = torch.stack(tList, dim=0)
            h1 = self.act(self.M[i](ActionEmb[:,i,:].squeeze()) + tt)
            # h1 = self.M[i](ActionEmb[:,i,:].squeeze()) + tt
            h0 = h1
            p0 = p1
        return self.T(h0)