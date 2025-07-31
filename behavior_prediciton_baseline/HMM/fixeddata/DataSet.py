import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class FixedLengthActionOnlyDataset(Dataset):
    def __init__(self, file_path, sequence_length=40):
        """
        Action-only, fixed-length dataset
        Args:
            file_path: Path to data file
            sequence_length: Sequence length, default 40 (10 quadruples)
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.data = []
        # Load raw data
        raw_data = pickle.load(open(self.file_path, 'rb'))
        # Keep only sequences of length 40
        for sequence in raw_data:
            if len(sequence) == sequence_length:
                # Extract action column (the 4th number in each quadruple)
                actions = []
                for i in range(0, sequence_length, 4):
                    if i + 3 < sequence_length:
                        action = sequence[i + 3]
                        actions.append(action)
                if len(actions) == 10:
                    self.data.append(actions)
        print(f"Loaded {len(self.data)} fixed-length action-only sequences from {file_path}")
        if len(self.data) > 0:
            print(f"Each sequence has {len(self.data[0])} actions")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_actions = sequence[:-1]  # First 9 actions
        target_action = sequence[-1]   # 10th action
        return (
            torch.tensor(input_actions, dtype=torch.long),
            torch.tensor(target_action, dtype=torch.long)
        )

def collate_fn_fixed(batch):
    input_actions_list, targets = zip(*batch)
    input_actions = torch.stack(input_actions_list, dim=0)  # [batch_size, 9]
    targets = torch.tensor(targets, dtype=torch.long)       # [batch_size]
    return input_actions, targets 