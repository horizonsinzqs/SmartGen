import random
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models1 import TransformerAutoencoder, TimeSeriesDataset1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pad(vocab_size, sequences):
    for sequence in sequences:
        if len(sequence) < 40:
            sequence.extend([vocab_size - 1] * (40 - len(sequence)))
    return sequences


def make_data(vocab_size, data_file='reduced_flattened_useful_us_trn_instance_10.pkl', batch_size=512):
    with open(data_file, 'rb') as file:
        sequence = pickle.load(file)
    data = pad(vocab_size, sequence)

    data = np.array(data)

    dataset = TimeSeriesDataset1(vocab_size, data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask):

        src_emb = self.embedding(src)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        tgt_emb = src_emb

        output = self.decoder(
            tgt_emb, memory,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=src_key_padding_mask
        )

        return self.output_layer(output)


def Train(dataset, ori_env, vocab_size):
    setup_seed(2024)

    num_epochs = 15
    seq_len = 40

    model = TransformerAutoencoder(vocab_size, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    model = model.to(device)
    train_file = f"IoT_data/{dataset}/{ori_env}/split_trn.pkl"

    train_loader = make_data(vocab_size, data_file=train_file)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters())
    model_name = f"IoT_model/Transformer_{dataset}_{ori_env}_{num_epochs}epoch.pth"

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            src, padding_mask, mask_v = batch
            src = src.to(device)
            mask_v = mask_v.to(device)
            padding_mask = padding_mask.to(device)

            output = model(src, src_key_padding_mask=padding_mask)
            src = src.cuda().long()

            loss = criterion(output.view(-1, vocab_size), src.view(-1))
            loss = loss.reshape(-1, seq_len) * mask_v
            loss = torch.sum(loss) / torch.sum(mask_v)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), model_name)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch:{epoch}, Loss:{avg_loss}")
