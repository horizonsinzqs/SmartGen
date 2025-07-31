import time
import DataSet
import torch
from torch import nn
import torch.optim as optim
import datetime
import os
import sys

from tqdm import tqdm
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = 'us'


def evaluteTop135(model, loader):
    model.eval()
    c1 = 0
    c3 = 0
    c5 = 0
    total = len(loader.dataset)
    for X, q1, q2, y in loader:
        X, q1, q2, y = X.to(device), q1.to(device), q2.to(device), y.to(device)
        with torch.no_grad():
            day = X[:, :, 0]
            tim = X[:, :, 1]
            logits = model(X, day, tim)
            max1 = max((1,1))
            max3 = max((1, 3))
            max5 = max((1, 5))
            y_resize = y.view(-1, 1)
            _1, pred1 = logits.topk(max1, 1, True, True)
            _3, pred3 = logits.topk(max3, 1, True, True)
            _5, pred5 = logits.topk(max5, 1, True, True)
            c1 += torch.eq(pred1, y_resize).sum().float().item()
            c3 += torch.eq(pred3, y_resize).sum().float().item()
            c5 += torch.eq(pred5, y_resize).sum().float().item()
    return c1 / total, c3 / total, c5 / total

def evaluateTopK(predict, target, k):
    ck = 0
    total = target.shape[0]
    for t in range(predict.shape[0]):
        predictY, Y = predict[t].reshape(1, -1), target[t]
        maxk = max((1, k))
        y_resize = Y.reshape(-1, 1)
        _k, predk = predictY.topk(maxk, dim=1, largest=True, sorted=True) # topk return values, indices
        ck += torch.eq(predk, y_resize).sum().float().item()
    return ck / total

def MAP(predict, target, k):
    ck = 0
    total = target.shape[0]
    for t in range(predict.shape[0]):
        predictY, Y = predict[t].reshape(1, -1), target[t]
        maxk = max((1, k))
        y_resize = Y.reshape(-1, 1)
        _k, predk = predictY.topk(maxk, 1, True, True)
        c1 = 0.0
        for i in range(_k.shape[-1]):
            if predk[0][i] == y_resize:
                c1 += 1 / (i + 1)
        ck += c1
    return ck / total

def precision(predict, target, k):
    tp = 0
    total = k * predict.shape[0]
    for t in range(predict.shape[0]):
        predictY, Y = predict[t].reshape(1, -1), target[t]
        maxk = max((1, k))
        y_resize = Y.reshape(-1, 1)
        _k, predk = predictY.topk(maxk, 1, True, True)
        tp += torch.eq(predk, y_resize).sum().float().item()
    return tp / total

def recall(predict, target, k):
    tp = 0
    total = predict.shape[0]
    for t in range(predict.shape[0]):
        predictY, Y = predict[t].reshape(1, -1), target[t]
        maxk = max((1, k))
        y_resize = Y.reshape(-1, 1)
        _k, predk = predictY.topk(maxk, 1, True, True)
        tp += torch.eq(predk, y_resize).sum().float().item()
    return tp / total

def F1Score(predict, target, k):
    _precision = precision(predict, target, k)
    _recall = recall(predict, target, k)
    return _precision * _recall / (_precision + _recall) * 2