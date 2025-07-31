import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from models1 import TransformerAutoencoder, TimeSeriesDataset1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}


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
    for i in range(len(sequences)):
        if len(sequences[i]) < 40:
            sequences[i].extend([vocab_size - 1] * (40 - len(sequences[i])))
        elif len(sequences[i]) > 40:
            sequences[i] = sequences[i][:40]

    return sequences


def split_random(data_file, train_file, vld_file, split_ratio=0.8, seed=2024):
    with open(data_file, 'rb') as file:
        data = pickle.load(file)
    random.seed(seed)

    split_index = int(len(data) * split_ratio)

    random.shuffle(data)

    dataset_1 = data[:split_index]
    dataset_2 = data[split_index:]

    with open(train_file, 'wb') as f3:
        pickle.dump(dataset_1, f3)

    with open(vld_file, 'wb') as f3:
        pickle.dump(dataset_2, f3)


def calculate_variance(numbers):
    mean = sum(numbers) / len(numbers)

    squared_differences = [(x - mean) ** 2 for x in numbers]

    variance = sum(squared_differences) / len(numbers)

    return variance


def detect_outliers_iqr(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    upper_bound = q3 + 1.5 * iqr

    non_outlier_indices = np.where((data <= upper_bound))[0]
    outlier_indices = np.where((data > upper_bound))[0]

    return non_outlier_indices.tolist(), outlier_indices.tolist()


def filter_by_indices(nested_list, indices):
    return [nested_list[i] for i in indices if i < len(nested_list)]


def save_outliers(outliers_seq, outlier_file):
    if len(outliers_seq) >= 1:
        for i in range(len(outliers_seq)):
            oseq = [outliers_seq[i]]
            with open(f'{outlier_file}_{i}.pkl', 'wb') as file4:
                pickle.dump(oseq, file4)
        return 1
    else:
        return 0


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--epochs', default=15, type=int)

    parser.add_argument('--model', default='TransformerAutoencoder', type=str, metavar='MODEL',
                        help='Name of model to train: Autoencoder/GRUAutoencoder/TransformerAutoencoder/MaskedAutoencoder')
    parser.add_argument('--dataset', default='fr', type=str, metavar='MODEL',
                        help='Name of dataset to train: an/fr/us/sp')

    parser.add_argument('--mask_strategy', default='top_k_loss', type=str, metavar='MODEL',
                        help='Mask strategy:random/top_k_loss/loss_guided')
    return parser


def make_data(new_env, vocab_size, data_file='reduced_flattened_useful_us_trn_instance_10.pkl', batch_size=32):
    with open(data_file, 'rb') as file:
        sequences = pickle.load(file)

    data = pad(vocab_size, sequences)
    data = np.array(data)

    dataset = TimeSeriesDataset1(vocab_size, data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def make_data_check(new_env, vocab_size, trn_file, add_file, batch_size=32):
    with open(trn_file, 'rb') as file:
        sequence = pickle.load(file)
    with open(add_file, 'rb') as file2:
        add_sequences = pickle.load(file2)
    sequences = sequence + add_sequences

    data = pad(vocab_size, sequences)
    data = np.array(data)

    dataset = TimeSeriesDataset1(vocab_size, data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def train(new_env, vocab_size, epochs, train_file, model_name, seq_len):
    model = TransformerAutoencoder(vocab_size=vocab_size, d_model=512, nhead=8, num_encoder_layers=2,
                                   num_decoder_layers=2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = epochs

    train_loader = make_data(new_env, vocab_size, data_file=train_file)

    best_val_loss = 1000
    stop_count = 0
    early_stop_count = 10

    last_loss_vector = {}
    last_number_vector = {}
    res = []
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_all = 0

        loss_vector = {}
        number_vector = {}
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

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")

    print('Finished Training')


def train_check(new_env, vocab_size, epochs, train_file, add_file, model_name, seq_len):
    model = TransformerAutoencoder(vocab_size=vocab_size, d_model=512, nhead=8, num_encoder_layers=2,
                                   num_decoder_layers=2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = epochs

    train_loader = make_data_check(new_env, vocab_size, train_file, add_file)

    best_val_loss = 1000
    stop_count = 0
    early_stop_count = 10

    last_loss_vector = {}
    last_number_vector = {}
    res = []
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_all = 0

        loss_vector = {}
        number_vector = {}
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

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")

    print('Finished Training')


def vld_check(new_env, vocab_size, vld_file, model_name, seq_len):
    val_loader = make_data(new_env, vocab_size, data_file=vld_file, batch_size=1)
    model = TransformerAutoencoder(vocab_size, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2)

    criterion = nn.CrossEntropyLoss(reduction='none')
    model.load_state_dict(torch.load(model_name))
    model.eval()

    losses = []
    model.to(device)
    total_loss = 0
    for batch in val_loader:
        src, padding_mask, mask_v = batch
        src = src.to(device)
        mask_v = mask_v.to(device)
        padding_mask = padding_mask.to(device)

        output = model(src, src_key_padding_mask=padding_mask)
        src = src.cuda().long()

        loss = criterion(output.view(-1, vocab_size), src.view(-1))
        loss = loss.reshape(-1, seq_len) * mask_v
        loss = torch.sum(loss) / torch.sum(mask_v)

        losses.append(loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)

    return avg_loss


def check_outlier(new_env, vocab_size, data_file, save_file, model_name, seq_len):
    val_loader = make_data(new_env, vocab_size, data_file=data_file, batch_size=1)
    model = TransformerAutoencoder(vocab_size, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2)

    criterion = nn.CrossEntropyLoss(reduction='none')
    model.load_state_dict(torch.load(model_name))
    model.eval()

    losses = []
    model.to(device)
    for batch in val_loader:
        src, padding_mask, mask_v = batch
        src = src.to(device)
        mask_v = mask_v.to(device)
        padding_mask = padding_mask.to(device)

        output = model(src, src_key_padding_mask=padding_mask)
        src = src.cuda().long()

        loss = criterion(output.view(-1, vocab_size), src.view(-1))
        loss = loss.reshape(-1, seq_len) * mask_v
        loss = torch.sum(loss) / torch.sum(mask_v)

        losses.append(loss.item())

    non_outlier_list, outlier_list = detect_outliers_iqr(losses)
    with open(data_file, 'rb') as file3:
        generated_sequence = pickle.load(file3)
    with open(data_file, 'rb') as file7:
        generated_sequence_2 = pickle.load(file7)
    l_g = len(generated_sequence)
    reversed_sequence = filter_by_indices(generated_sequence, non_outlier_list)
    outlier_sequence = filter_by_indices(generated_sequence_2, outlier_list)
    l_r = len(reversed_sequence)

    with open(save_file, 'wb') as file4:
        pickle.dump(reversed_sequence, file4)

    print(
        f'The reversed_sequences ： {reversed_sequence}. \n The number of original generated sequences is {l_g}. \n The number of reversed sequences is {l_r}.')

    return reversed_sequence, outlier_sequence


def security_check(dataset, new_env, thres, method, model):
    model_name = f"check_model/best_{dataset}_{model}_{method}.pth"
    vocab_size = vocab_dic[dataset]
    epochs = 10
    seq_len = 40
    data_file = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq.pkl'
    save_file = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter.pkl'

    setup_seed(2024)
    train(new_env, vocab_size, epochs, data_file, model_name, seq_len)
    reversed_sequence, outlier_sequence = check_outlier(new_env, vocab_size, data_file, save_file, model_name, seq_len)

    outlier_file = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter_out'
    flag = save_outliers(outlier_sequence, outlier_file)

    if flag == 1:
        print('============================NEED FIND TRUE OUTLIER================================ \n')
        trn_file = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter_trn.pkl'
        vld_file = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter_vld.pkl'
        split_random(save_file, trn_file, vld_file)

        train(new_env, vocab_size, epochs, trn_file, model_name, seq_len)
        standard_loss = vld_check(new_env, vocab_size, vld_file, model_name, seq_len)

        for i in range(len(outlier_sequence)):
            add_file = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter_out_{i}.pkl'
            train_check(new_env, vocab_size, epochs, trn_file, add_file, model_name, seq_len)
            new_loss = vld_check(new_env, vocab_size, vld_file, model_name, seq_len)
            if new_loss <= standard_loss:
                print(f'-------------------------------------------- THIS ONE NEED REVERSE: {outlier_sequence[i]}')
                reversed_sequence = reversed_sequence + [outlier_sequence[i]]
            else:
                print(f'--------------------------------------------- THIS ONE NEED OUT: {outlier_sequence[i]}.')

    true_save_file = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter_true.pkl'
    with open(data_file, 'rb') as file:
        sequences = pickle.load(file)
    if len(reversed_sequence) == len(sequences):
        with open(true_save_file, 'wb') as file5:
            pickle.dump(sequences, file5)
    else:
        with open(true_save_file, 'wb') as file9:
            pickle.dump(reversed_sequence, file9)

    print(
        f'The true reversed sequences ： {reversed_sequence}. \n The true number of reversed sequences is {len(reversed_sequence)}.')
