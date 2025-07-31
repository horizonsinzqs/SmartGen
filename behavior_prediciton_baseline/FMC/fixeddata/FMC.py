# -*- coding: utf-8 -*-
"""
Modified FPMC (Factorized Personalized Markov Chain) Model
Supports multi-feature input and variable-length sequence processing
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FPMC(nn.Module):
    def __init__(self, actionNum=300, deviceNum=50, k_embedding=64):
        """
        FPMC model, supports variable-length sequences and full timestep information
        Args:
            actionNum: Number of actions
            deviceNum: Number of devices
            k_embedding: Embedding dimension
        """
        super(FPMC, self).__init__()
        print("=" * 10, "Creating Enhanced FPMC Model", "=" * 10)
        
        self.actionNum = actionNum
        self.deviceNum = deviceNum
        self.k_embedding = k_embedding
        
        # Embedding layers
        self.action_emb = nn.Embedding(actionNum + 1, k_embedding, padding_idx=0)
        self.device_emb = nn.Embedding(deviceNum + 1, k_embedding, padding_idx=0)
        self.day_emb = nn.Embedding(8, k_embedding)  # Weekday 0-6 + padding
        self.time_emb = nn.Embedding(9, k_embedding)  # Time slot 0-7 + padding
        
        # FPMC core embeddings
        # IL: Item-Last item interaction (action transition)
        self.IL = nn.Embedding(actionNum + 1, k_embedding, padding_idx=0)
        self.LI = nn.Embedding(actionNum + 1, k_embedding, padding_idx=0)
        
        # Device transition embeddings
        self.DL = nn.Embedding(deviceNum + 1, k_embedding, padding_idx=0)
        self.LD = nn.Embedding(deviceNum + 1, k_embedding, padding_idx=0)
        
        # Feature fusion layer
        self.feature_fusion = nn.Linear(k_embedding * 4, k_embedding)
        
        # Time-aware weights
        self.time_attention = nn.Linear(k_embedding * 2, 1)
        
        # Output layer
        self.output_layer = nn.Linear(k_embedding + 2, actionNum)
        
        # Device to action projection layer
        self.device_to_action = nn.Linear(k_embedding, actionNum)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, days, times, devices, actions, masks):
        """
        Forward pass
        Args:
            days: [batch_size, seq_len] weekday feature
            times: [batch_size, seq_len] time feature
            devices: [batch_size, seq_len] device feature
            actions: [batch_size, seq_len] action feature
            masks: [batch_size, seq_len] mask
        """
        batch_size, seq_len = actions.shape
        
        # Feature embeddings
        day_emb = self.day_emb(days)        # [batch_size, seq_len, k_embedding]
        time_emb = self.time_emb(times)     # [batch_size, seq_len, k_embedding]
        device_emb = self.device_emb(devices)  # [batch_size, seq_len, k_embedding]
        action_emb = self.action_emb(actions)  # [batch_size, seq_len, k_embedding]
        
        # Feature fusion
        combined_features = torch.cat([day_emb, time_emb, device_emb, action_emb], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # [batch_size, seq_len, k_embedding]
        fused_features = self.dropout(fused_features)
        
        # Compute sequence representation
        sequence_repr = self._compute_sequence_representation(fused_features, actions, devices, masks)
        
        # Compute embedding of last action
        last_actions = self._get_last_valid_items(actions, masks)  # [batch_size]
        last_devices = self._get_last_valid_items(devices, masks)  # [batch_size]
        
        # FPMC core: action transition pattern
        action_transition = self._compute_action_transition(last_actions)  # [batch_size, actionNum]
        
        # Device transition pattern
        device_transition = self._compute_device_transition(last_devices)  # [batch_size, actionNum]
        
        # Combine features - handle dimensions correctly
        # Convert transition scores to feature representation
        action_transition_mean = action_transition.mean(dim=1, keepdim=True)  # [batch_size, 1]
        device_transition_mean = device_transition.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Expand to match sequence representation dimension
        action_transition_expanded = action_transition_mean.unsqueeze(-1).expand(-1, sequence_repr.size(1), 1)  # [batch_size, seq_len, 1]
        device_transition_expanded = device_transition_mean.unsqueeze(-1).expand(-1, sequence_repr.size(1), 1)  # [batch_size, seq_len, 1]
        
        # Combine features
        combined_repr = torch.cat([
            sequence_repr,  # [batch_size, seq_len, k_embedding]
            action_transition_expanded,  # [batch_size, seq_len, 1]
            device_transition_expanded   # [batch_size, seq_len, 1]
        ], dim=-1)  # [batch_size, seq_len, k_embedding + 2]
        
        # Final prediction - average over sequence dimension
        sequence_averaged = combined_repr.mean(dim=1)  # [batch_size, k_embedding + 2]
        
        # Directly use action and device transition scores
        final_features = torch.cat([
            sequence_averaged,  # [batch_size, k_embedding + 2]
            action_transition,  # [batch_size, actionNum]
            device_transition   # [batch_size, actionNum]
        ], dim=-1)  # [batch_size, k_embedding + 2 + 2*actionNum]
        
        # Use a simpler method: weighted combination
        output = action_transition + device_transition + self.output_layer(sequence_averaged)  # [batch_size, actionNum]
        
        return output
    
    def _compute_sequence_representation(self, fused_features, actions, devices, masks):
        """Compute sequence representation, considering time weights"""
        batch_size, seq_len, k_embedding = fused_features.shape
        
        # Time decay weights
        position_weights = []
        for i in range(batch_size):
            valid_length = int(masks[i].sum().item())
            if valid_length > 0:
                # Give higher weights to more recent elements
                weights = torch.softmax(torch.arange(valid_length, dtype=torch.float) + 1, dim=0).to(device)
                padded_weights = torch.cat([
                    weights, 
                    torch.zeros(seq_len - valid_length).to(device)
                ])
            else:
                padded_weights = torch.zeros(seq_len).to(device)
            position_weights.append(padded_weights)
        
        position_weights = torch.stack(position_weights).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Weighted sequence representation
        weighted_features = fused_features * position_weights * masks.unsqueeze(-1)
        
        return weighted_features
    
    def _compute_action_transition(self, last_actions):
        """Compute action transition scores"""
        batch_size = last_actions.size(0)
        
        # Get all possible next actions
        all_actions = torch.arange(1, self.actionNum + 1).to(device)  # [actionNum]
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)  # [batch_size, actionNum]
        
        # Compute transition scores
        last_action_emb = self.LI(last_actions).unsqueeze(1)  # [batch_size, 1, k_embedding]
        next_action_emb = self.IL(all_actions)  # [batch_size, actionNum, k_embedding]
        
        # Compute similarity
        transition_scores = torch.sum(last_action_emb * next_action_emb, dim=-1)  # [batch_size, actionNum]
        
        return transition_scores
    
    def _compute_device_transition(self, last_devices):
        """Compute device transition scores"""
        batch_size = last_devices.size(0)
        
        # Get last device embedding
        last_device_emb = self.LD(last_devices)  # [batch_size, k_embedding]
        
        # Use device embedding to predict action distribution
        device_scores = self.device_to_action(last_device_emb)  # [batch_size, actionNum]
        
        return device_scores
    
    def _get_last_valid_items(self, items, masks):
        """Get the last valid element of each sequence"""
        batch_size = items.size(0)
        last_items = []
        
        for i in range(batch_size):
            valid_length = int(masks[i].sum().item())
            if valid_length > 0:
                last_items.append(items[i, valid_length - 1])
            else:
                last_items.append(torch.tensor(0).to(device))  # padding index
        
        return torch.stack(last_items)