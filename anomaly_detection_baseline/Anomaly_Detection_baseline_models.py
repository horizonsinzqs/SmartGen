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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        device_control = sample.reshape(10, 4).T[3]

        encoder_input = device_control
        decoder_output = device_control

        encoder_input = torch.from_numpy(encoder_input)
        decoder_output = torch.from_numpy(decoder_output)

        return encoder_input, decoder_output


class TimeSeriesDataset2(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        device_control = sample.reshape(10, 4).T[1]

        encoder_input = device_control
        decoder_output = device_control

        encoder_input = torch.from_numpy(encoder_input)
        decoder_output = torch.from_numpy(decoder_output)

        return encoder_input, decoder_output


class TimeSeriesDataset3(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        device_control = sample.reshape(10, 4).T[2]

        encoder_input = device_control
        decoder_output = device_control

        encoder_input = torch.from_numpy(encoder_input)
        decoder_output = torch.from_numpy(decoder_output)

        return encoder_input, decoder_output


class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerAutoencoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src)

        output = self.transformer(src, src)

        output = self.fc(output)

        return output
