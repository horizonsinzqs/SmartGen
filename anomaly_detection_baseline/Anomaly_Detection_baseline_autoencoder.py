import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from Anomaly_Detection_baseline_models import GRUAutoencoder, TransformerAutoencoder, Autoencoder, TimeSeriesDataset1, \
    TimeSeriesDataset2, TimeSeriesDataset3

vocab_dic = {"an": 141, "fr": 222, "us": 268, "sp": 234}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    torch.use_deterministic_algorithms(True)


def make_data(args, data_file, batch_size=32):
    with open(data_file, 'rb') as file:
        data = pickle.load(file)
    data = np.array(data)

    if args.env == 'winter':
        dataset = TimeSeriesDataset1(data)
    elif args.env == 'daytime':
        dataset = TimeSeriesDataset2(data)
    elif args.env == 'single':
        dataset = TimeSeriesDataset3(data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def train(args, train_file):
    model_name = f"saved_model/best_{args.model}_{args.dataset}_{args.env}.pth"

    if args.model == "GRUAutoencoder":
        model = GRUAutoencoder(vocab_size=vocab_dic[args.dataset], input_size=128, hidden_size1=256, hidden_size2=64,
                               dropout_value=0.3)
    elif args.model == "Autoencoder":
        model = Autoencoder(vocab_size=vocab_dic[args.dataset], input_size=128, hidden_size1=256, hidden_size2=64)
    elif args.model == "TransformerAutoencoder":
        model = TransformerAutoencoder(vocab_size=vocab_dic[args.dataset], d_model=512, nhead=8, num_encoder_layers=2,
                                       num_decoder_layers=2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 15
    train_loader = make_data(args, data_file=train_file)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_batch, target_batch = batch
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            output = model(input_batch)
            outputs = output.view(-1, vocab_dic[args.dataset])
            target_batch = target_batch.view(-1)
            target_batch = target_batch.to(dtype=torch.long)

            loss = criterion(outputs, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), model_name)


def find_threshold(args, vld_file, percentage=95):
    model_name = f"saved_model/best_{args.model}_{args.dataset}_{args.env}.pth"
    val_loader = make_data(args, data_file=vld_file, batch_size=1)
    if args.model == "GRUAutoencoder":
        model = GRUAutoencoder(vocab_size=vocab_dic[args.dataset], input_size=128, hidden_size1=256, hidden_size2=64,
                               dropout_value=0.3)
    elif args.model == "Autoencoder":
        model = Autoencoder(vocab_size=vocab_dic[args.dataset], input_size=128, hidden_size1=256, hidden_size2=64)
    elif args.model == "TransformerAutoencoder":
        model = TransformerAutoencoder(vocab_dic[args.dataset], d_model=512, nhead=8, num_encoder_layers=2,
                                       num_decoder_layers=2)

    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    losses = []
    model.to(device)
    total_loss = 0
    for batch in val_loader:
        input_batch, target_batch = batch
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        outputs = model(input_batch)

        outputs = outputs.view(-1, vocab_dic[args.dataset])
        target_batch = target_batch.view(-1)
        target_batch = target_batch.to(dtype=torch.long)

        loss = criterion(outputs, target_batch)

        losses.append(loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Avg Loss (Validation Dataset): {avg_loss:.4f}")

    threshold = np.percentile(losses, percentage)
    print(f"Percentage:{percentage}% Threshold: {threshold}")
    return threshold


def evaluate(args, test_file2, threshold):
    model_name = f"saved_model/best_{args.model}_{args.dataset}_{args.env}.pth"
    with open(f"baseline_data/{args.dataset}/attack/labeled_{args.dataset}_{args.singleness_attack}.pkl", 'rb') as file:
        input_train_attack = pickle.load(file)

    with open(test_file2, 'rb') as file:
        tmp = pickle.load(file)
        input_train_test = []
        for item in tmp:
            input_train_test.append((item, 0))

    input_train_tuple = input_train_test + input_train_attack
    input_train = [item[0] for item in input_train_tuple]

    if args.model == "GRUAutoencoder":
        model = GRUAutoencoder(vocab_size=vocab_dic[args.dataset], input_size=128, hidden_size1=256, hidden_size2=64,
                               dropout_value=0.3)
    elif args.model == "Autoencoder":
        model = Autoencoder(vocab_size=vocab_dic[args.dataset], input_size=128, hidden_size1=256, hidden_size2=64)
    elif args.model == "TransformerAutoencoder":
        model = TransformerAutoencoder(vocab_dic[args.dataset], d_model=512, nhead=8, num_encoder_layers=2,
                                       num_decoder_layers=2)

    model.to(device)

    input_train = np.array(input_train)

    batch_size = 1
    if args.env == 'winter':
        test_dataset = TimeSeriesDataset1(input_train)
    elif args.env == 'daytime':
        test_dataset = TimeSeriesDataset2(input_train)
    elif args.env == 'single':
        test_dataset = TimeSeriesDataset3(input_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_name))
    model.eval()
    losses = []
    predictions = []

    total_loss = 0
    for batch in test_loader:
        input_batch, target_batch = batch
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        outputs = model(input_batch)

        outputs = outputs.view(-1, vocab_dic[args.dataset])
        target_batch = target_batch.view(-1)
        target_batch = target_batch.to(dtype=torch.long)

        loss = criterion(outputs, target_batch)

        losses.append(loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Avg Loss (Test Dataset): {avg_loss:.4f}")

    for i in range(len(losses)):
        if losses[i] < threshold:
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions


def anomaly_detection(args, train_file1, test_file2):
    setup_seed(2024)
    vld_file = f"baseline_data/{args.dataset}/{args.env}/vld.pkl"
    train(args, train_file1)
    threshold = find_threshold(args, vld_file, percentage=90)
    predictions = evaluate(args, test_file2, threshold=threshold)

    return predictions
