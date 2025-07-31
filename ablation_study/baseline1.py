import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from torch import optim
from torch.utils.data import DataLoader

from models1 import TransformerAutoencoder, TimeSeriesDataset2, TimeSeriesDataset3, TimeSeriesDataset4

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

    return dataset_1, dataset_2


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

    if new_env == 'spring':
        dataset = TimeSeriesDataset2(vocab_size, data)
    elif new_env == 'night':
        dataset = TimeSeriesDataset3(vocab_size, data)
    elif new_env == 'multiple':
        dataset = TimeSeriesDataset4(vocab_size, data)
    else:
        dataset = None

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

    print('Finished Training')


def find_threshold(new_env, vocab_size, vld_file, model_name, seq_len, percentage):
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
    print(f"Avg Loss (Validation Dataset): {avg_loss:.4f}")

    threshold = np.percentile(losses, percentage)
    print(f"Percentage:{percentage}% Threshold: {threshold}")
    return threshold


def evaluate(new_env, vocab_size, test_file1, test_file3, model_name, seq_len, threshold):
    with open(test_file1, 'rb') as file:
        input_train_attack1 = pickle.load(file)

    input_train_attack = input_train_attack1

    with open(test_file3, 'rb') as file:
        tmp = pickle.load(file)
        tmp = pad(vocab_size, tmp)
        input_train_test = []
        for item in tmp:
            input_train_test.append((item, 0))

    input_train_tuple = input_train_test + input_train_attack

    input_train = [item[0] for item in input_train_tuple]
    pad(vocab_size, input_train)
    labels = [item[1] for item in input_train_tuple]
    model = TransformerAutoencoder(vocab_size, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2)
    model.to(device)

    input_train = np.array(input_train)

    batch_size = 1
    if new_env == 'spring':
        test_dataset = TimeSeriesDataset2(vocab_size, input_train)
    elif new_env == 'night':
        test_dataset = TimeSeriesDataset3(vocab_size, input_train)
    elif new_env == 'multiple':
        test_dataset = TimeSeriesDataset4(vocab_size, input_train)
    else:
        test_dataset = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(reduction='none')
    model.load_state_dict(torch.load(model_name))
    model.eval()
    losses = []
    predictions = []
    FP_samples = []

    total_loss = 0
    for batch in test_loader:
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

    avg_loss = total_loss / len(test_loader)
    print(f"Avg Loss (Test Dataset): {avg_loss:.4f}")

    for i in range(len(losses)):
        if losses[i] < threshold:
            predictions.append(0)
        else:
            predictions.append(1)

    cm = confusion_matrix(y_true=labels, y_pred=predictions)

    for q in range(len(labels)):
        if labels[q] == 1 and predictions[q] == 0:
            FP_samples.append(input_train[q])

    TN, FP, FN, TP = cm.ravel()

    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)

    recall = recall_score(y_pred=predictions, y_true=labels)
    precision = precision_score(y_pred=predictions, y_true=labels)
    accuracy = accuracy_score(y_pred=predictions, y_true=labels)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)
    print("False Positive Rate:", FPR)
    print("False Negative Rate:", FNR)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1_score)

    max_loss = max(losses)
    min_loss = min(losses)

    print(f"Max loss {max_loss:.4f} , Min loss: {min_loss:.4f}")
    print('Finished Test')

    return TP, TN, FP, FN, FPR, FNR, recall, precision, accuracy, f1_score


def Anomaly_detection(dataset, new_env, thres, method, model, study, per, type):
    model_name = f"check_model/best_{study}_{dataset}_{new_env}_generation_{method}_th={thres}_{model}_model_{type}.pth"
    vocab_size = vocab_dic[dataset]
    epochs = 15
    seq_len = 10

    if type == 'ori':
        data_file = f'study_data/{study}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq.pkl'
    elif type == 'S1':
        data_file = f'study_data/{study}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter.pkl'
    elif type == 'S2':
        data_file = f'study_data/{study}/{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter_true.pkl'

    if new_env == 'multiple':
        vld_file = data_file
        train_file = data_file
    else:
        train_file = f'test/trn.pkl'
        vld_file = f"test/rs_vld.pkl"
        split_random(data_file, train_file, vld_file)

    if new_env == 'spring':
        test_file1 = f"attack/{dataset}/labeled_{dataset}_spring_attack_heater.pkl"
    elif new_env == 'night':
        test_file1 = f"attack/{dataset}/labeled_{dataset}_night_attack_time.pkl"
    elif new_env == 'multiple':
        test_file1 = f"attack/{dataset}/labeled_{dataset}_multiple_attack_tv.pkl"

    test_file3 = f"test/{dataset}/{new_env}/split_test.pkl"

    setup_seed(2024)
    train(new_env, vocab_size, epochs, train_file, model_name, seq_len)
    threshold = find_threshold(new_env, vocab_size, vld_file, model_name, seq_len, percentage=per)
    TP, TN, FP, FN, FPR, FNR, recall, precision, accuracy, f1_score = evaluate(new_env, vocab_size, test_file1,
                                                                               test_file3, model_name, seq_len,
                                                                               threshold=threshold)
    result = {
        f'recall_{type}': recall,
        f'precision_{type}': precision,
        f'F1 score_{type}': f1_score,
    }
    return result


def Anomaly_detection_ab1(dataset, new_env, thres, model, study, per, module):
    model_name = f"check_model/best_{study}_{module}_{dataset}_{new_env}_generation_th={thres}_{model}_model.pth"
    vocab_size = vocab_dic[dataset]
    epochs = 15
    seq_len = 10

    data_file = f'study_data/{study}/generation_th={thres}_{model}_seq_{module}.pkl'

    vld_file = f'study_data/{study}/split_vld.pkl'

    test_file1 = f"study_data/{study}/labeled_sp_spring_attack_heater.pkl"

    test_file3 = f"study_data/{study}/split_test.pkl"
    if module == 'without_all':
        per = 68

    setup_seed(2024)
    train(new_env, vocab_size, epochs, data_file, model_name, seq_len)
    threshold = find_threshold(new_env, vocab_size, vld_file, model_name, seq_len, percentage=per)
    TP, TN, FP, FN, FPR, FNR, recall, precision, accuracy, f1_score = evaluate(new_env, vocab_size, test_file1,
                                                                               test_file3, model_name, seq_len,
                                                                               threshold=threshold)
    result = {
        'recall': recall,
        'precision': precision,
        'F1 score': f1_score,
    }
    return result
