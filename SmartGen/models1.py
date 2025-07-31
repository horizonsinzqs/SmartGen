import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Autoencoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size1, hidden_size2):
        super(Autoencoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_size)

        self.encoder_1 = nn.Linear(input_size, hidden_size1)
        self.encoder_2 = nn.Linear(hidden_size1, hidden_size2)

        self.decoder_1 = nn.Linear(hidden_size2, hidden_size1)
        self.decoder_2 = nn.Linear(hidden_size1, input_size)

        self.decoder_output = nn.Linear(input_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder_1(x)
        x = self.encoder_2(x)

        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_output(x)
        return x


class GRUAutoencoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size1, hidden_size2, dropout_value):
        super(GRUAutoencoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_size)

        self.encoder_rnn1 = nn.GRU(input_size, hidden_size1, dropout=dropout_value, batch_first=True)
        self.encoder_rnn2 = nn.GRU(hidden_size1, hidden_size2, dropout=dropout_value, batch_first=True)

        self.decoder_rnn1 = nn.GRU(hidden_size2, hidden_size1, dropout=dropout_value, batch_first=True)
        self.decoder_rnn2 = nn.GRU(hidden_size1, input_size, dropout=dropout_value, batch_first=True)

        self.decoder_output = nn.Linear(input_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.encoder_rnn1(x)
        x, _ = self.encoder_rnn2(x)

        x, _ = self.decoder_rnn1(x)
        x, _ = self.decoder_rnn2(x)
        x = self.decoder_output(x)
        return x


def find_next_occurrence_at_indices(arr, target, indices, i):
    first_occurrence = None

    for t in indices:
        if 0 <= t < len(arr) and arr[t] == target:
            first_occurrence = t + i + 1
            break
        first_occurrence = indices[-1]

    return first_occurrence


def _compute_duration_embeddings(input_ids: torch.Tensor) -> torch.Tensor:
    embedding_dim = 512
    persistence_indices = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
    seq_len = 40
    duration_embeddings = torch.zeros(int(seq_len / 4), device=input_ids.device)
    time_embeddings = torch.zeros(int(seq_len / 4), device=input_ids.device)
    day_embeddings = input_ids[0:37:4].tolist()
    hour_embeddings = input_ids[1:38:4].tolist()
    time_embeddings[0] = 0
    for i in range(len(day_embeddings) - 1):
        if day_embeddings[i] * 24 + hour_embeddings[i] * 3 <= day_embeddings[i + 1] * 24 + \
                hour_embeddings[i + 1] * 3:
            time_embeddings[i + 1] = day_embeddings[i + 1] * 24 + hour_embeddings[i + 1] * 3 - day_embeddings[i] * 24 - \
                                     hour_embeddings[i] * 3 + time_embeddings[i]
        else:
            time_embeddings[i + 1] = day_embeddings[i + 1] * 24 + hour_embeddings[i + 1] * 3 - day_embeddings[i] * 24 - \
                                     hour_embeddings[i] * 3 + 168 + time_embeddings[i]

    input_seq = input_ids
    behavior_list = []
    intro_time = []
    for idx in persistence_indices:
        behavior_list.append(input_seq[idx])
    indices = [t for t in range(0, len(behavior_list))]
    for i in range(len(behavior_list)):
        next_arr = behavior_list[i + 1:]
        next_index = find_next_occurrence_at_indices(next_arr, behavior_list[i], indices, i)
        intro_time.append(next_index)

    for i in range(len(intro_time)):
        duration_embeddings[i] = time_embeddings[intro_time[i]] - time_embeddings[i]

    scaled_duration = (duration_embeddings.unsqueeze(1) * torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float, device=input_ids.device) * -(
                np.log(10000.0) / embedding_dim))).float()
    scaled_duration = torch.tensor(scaled_duration)
    sinusoidal_duration_embedding = torch.zeros(int(seq_len / 4), embedding_dim, device=input_ids.device)
    sinusoidal_duration_embedding[:, 0::2] = scaled_duration
    sinusoidal_duration_embedding[:, 1::2] = scaled_duration

    return sinusoidal_duration_embedding


class TimeSeriesDataset1(Dataset):
    def __init__(self, vocab_size, data):
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        device_control = sample
        mask = (device_control == self.vocab_size - 1)
        mask_v = (device_control != self.vocab_size - 1)

        encoder_input = device_control

        encoder_input = torch.from_numpy(encoder_input)

        return encoder_input, mask, mask_v


class TimeSeriesDataset2(Dataset):
    def __init__(self, vocab_size, data):
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        device_control = sample.reshape(10, 4).T[3]
        mask = (device_control == self.vocab_size - 1)
        mask_v = (device_control != self.vocab_size - 1)

        encoder_input = device_control

        encoder_input = torch.from_numpy(encoder_input)

        return encoder_input, mask, mask_v


class TimeSeriesDataset3(Dataset):
    def __init__(self, vocab_size, data):
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        device_control = sample.reshape(10, 4).T[1]
        mask = (device_control == self.vocab_size - 1)
        mask_v = (device_control != self.vocab_size - 1)

        encoder_input = device_control

        encoder_input = torch.from_numpy(encoder_input)

        return encoder_input, mask, mask_v


class TimeSeriesDataset4(Dataset):
    def __init__(self, vocab_size, data):
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        device_control = sample.reshape(10, 4).T[2]
        mask = (device_control == self.vocab_size - 1)
        mask_v = (device_control != self.vocab_size - 1)

        encoder_input = device_control

        encoder_input = torch.from_numpy(encoder_input)

        return encoder_input, mask, mask_v


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
