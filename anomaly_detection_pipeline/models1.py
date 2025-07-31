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
