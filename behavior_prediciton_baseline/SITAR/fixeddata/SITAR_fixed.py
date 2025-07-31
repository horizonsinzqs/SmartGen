import torch
import torch.nn as nn
from SITAR import STAR

def get_actiononly_input(input_actions, action_num):
    # input_actions: [batch, 9]
    # Construct [batch, 9, 4], last column is action, others are 0
    batch, seq_len = input_actions.shape
    x = torch.zeros((batch, seq_len, 4), dtype=torch.long, device=input_actions.device)
    x[:, :, 3] = input_actions
    return x

class SITAR_Fixed(nn.Module):
    def __init__(self, action_num, d_model=40, input_length=9):
        super(SITAR_Fixed, self).__init__()
        self.action_num = action_num
        self.d_model = d_model
        self.input_length = input_length
        self.star = STAR(sequenceLength=10, d_model=d_model, actionNum=action_num)
    def forward(self, input_actions):
        # input_actions: [batch, 9]
        x = get_actiononly_input(input_actions, self.action_num)  # [batch, 9, 4]
        # t1, t2 all zeros
        t1 = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.long, device=x.device)
        t2 = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.long, device=x.device)
        return self.star(x, t1, t2) 