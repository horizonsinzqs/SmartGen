import torch
import torch.nn as nn
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CARNN_fixed(nn.Module):
    def __init__(self, input_length=9, d_model=64, actionNum=300, max_sequence_length=50, timeStamp=7*8):
        """
        Fixed-length CARNN model, using only action column for training and prediction
        Args:
            input_length: Input sequence length, default 9 (first 9 actions)
            d_model: Embedding dimension
            actionNum: Number of actions
            max_sequence_length: Maximum sequence length
            timeStamp: Number of timestamps
        """
        super(CARNN_fixed, self).__init__()
        
        # Action embedding layer
        self.actionEmb = nn.Embedding(actionNum + 1, d_model, padding_idx=0)  # +1 for padding
        
        # CARNN core components
        self.M = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(max_sequence_length)])
        self.W = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(timeStamp)])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, actionNum)
        
        self.d_model = d_model
        self.input_length = input_length
        self.actionNum = actionNum
        self.max_sl = max_sequence_length
        self.timeStamp = timeStamp

    def forward(self, input_actions):
        """
        Forward pass
        Args:
            input_actions: [batch_size, 9] input action sequence
        """
        batch_size, seq_len = input_actions.shape
        
        # Initialize hidden state
        hl = torch.zeros(batch_size, self.d_model).to(device)
        
        # Action embedding
        action_emb = self.actionEmb(input_actions)  # [batch_size, 9, d_model]
        
        # Process sequence
        for i in range(seq_len):
            if i < len(self.M):
                # Use sequence position as time difference approximation
                time_diff = torch.full((batch_size,), i % self.timeStamp, dtype=torch.long).to(device)
                
                # Calculate new hidden state
                new_hl = []
                for j in range(batch_size):
                    time_idx = int(time_diff[j].item())
                    if time_idx < len(self.W):
                        h_new = torch.sigmoid(
                            self.M[min(i, len(self.M)-1)](action_emb[j, i]) + 
                            self.W[time_idx](hl[j])
                        )
                    else:
                        h_new = torch.sigmoid(
                            self.M[min(i, len(self.M)-1)](action_emb[j, i]) + 
                            self.W[0](hl[j])
                        )
                    new_hl.append(h_new)
                
                hl = torch.stack(new_hl, dim=0)
        
        # Output prediction
        output = self.output_layer(hl)  # [batch_size, actionNum]
        
        return output 