import torch
import torch.nn as nn

class LSTM_Fixed(nn.Module):
    def __init__(self, action_num, d_model=64, input_length=9):
        super(LSTM_Fixed, self).__init__()
        self.action_emb = nn.Embedding(action_num + 1, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.fc = nn.Linear(d_model, action_num)
        self.input_length = input_length
        self.d_model = d_model
        self.action_num = action_num
    def forward(self, input_actions):
        # input_actions: [batch, 9]
        emb = self.action_emb(input_actions)  # [batch, 9, d_model]
        lstm_out, _ = self.lstm(emb)          # [batch, 9, d_model]
        out = lstm_out[:, -1, :]              # [batch, d_model]
        logits = self.fc(out)                 # [batch, action_num]
        return logits 