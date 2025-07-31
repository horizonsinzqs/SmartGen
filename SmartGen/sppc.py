import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from models1 import TransformerAutoencoder, TimeSeriesDataset1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad(vocab_size, sequences):
    for sequence in sequences:
        if len(sequence) < 40:
            sequence.extend([vocab_size - 1] * (40 - len(sequence)))
    return sequences


def remove_pad(lst):
    for sublist in lst:
        while sublist and sublist[-1] == 0:
            sublist.pop()
    return lst


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


def simi_pad(sequences):
    for sequence in sequences:
        if len(sequence) < 40:
            sequence.extend([0] * (40 - len(sequence)))
    return sequences


def make_data(vocab_size, data_file='reduced_flattened_useful_us_trn_instance_10.pkl', batch_size=64):
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

    def forward(self, src, src_key_padding_mask=None):
        src_emb = self.embedding(src)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        tgt_emb = src_emb

        return memory


def SPPC_select(dataset, ori_env, vocab_size, threshold):
    setup_seed(2024)
    num_epochs = 15
    for day in range(7):

        model = TransformerAutoencoder(vocab_size, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
        model = model.to(device)
        model_name = f"IoT_model/Transformer_{dataset}_{ori_env}_{num_epochs}epoch.pth"
        model.load_state_dict(torch.load(model_name))
        day_select_file = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}.pkl'

        with open(day_select_file, 'rb') as file2:
            text_collection = pickle.load(file2)
        print(len(text_collection))
        train_loader = make_data(vocab_size, data_file=day_select_file, batch_size=len(text_collection))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        for batch in train_loader:
            src, padding_mask, _ = batch
            src = src.to(device)
            padding_mask = padding_mask.to(device)
            memory = model(src, src_key_padding_mask=padding_mask)
            memories = memory.cpu().detach().numpy()
            memories_2D = memories.reshape(memories.shape[0], memories.shape[1] * memories.shape[2])

        similarity_matrix = cosine_similarity(memories_2D)

        to_remove = set()
        unique_indices = []

        for i in range(len(text_collection)):
            if i not in to_remove:
                unique_indices.append(i)
                for j in range(i + 1, len(text_collection)):
                    if similarity_matrix[i, j] > threshold:
                        to_remove.add(j)

        deduplicated_collection = [text_collection[i] for i in unique_indices]

        print(deduplicated_collection)
        print(len(deduplicated_collection))

        with open(f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_SPPC_th={threshold}.pkl', 'wb') as f3:
            pickle.dump(deduplicated_collection, f3)


def similarity_select(dataset, ori_env, threshold):
    for day in range(7):
        with open(f'IoT_data/{dataset}/{ori_env}/trn_day_{day}.pkl', 'rb') as file3:
            text_collection = pickle.load(file3)

        simi_pad(text_collection)
        similarity_matrix = cosine_similarity(text_collection)
        remove_pad(text_collection)
        to_remove = set()
        unique_indices = []

        for i in range(len(text_collection)):
            if i not in to_remove:
                unique_indices.append(i)
                for j in range(i + 1, len(text_collection)):
                    if similarity_matrix[i, j] > threshold:
                        to_remove.add(j)

        deduplicated_collection = [text_collection[i] for i in unique_indices]
        print(deduplicated_collection)
        print(len(deduplicated_collection))

        with open(f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_similarity_th={threshold}.pkl', 'wb') as f3:
            pickle.dump(deduplicated_collection, f3)
