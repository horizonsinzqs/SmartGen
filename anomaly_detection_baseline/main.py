from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import argparse
import random
import torch
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from models1 import GRUAutoencoder, TransformerAutoencoder, TimeSeriesDataset2, TimeSeriesDataset3, TimeSeriesDataset4
from collections import Counter
from Anomaly_Detection_baseline_autoencoder import anomaly_detection

vocab_dic = {"fr": 223, "sp": 235, "us": 269}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MarkovChain():
    def __init__(self, state_number):
        self.transition_matrix = None
        self.states = None
        self.state_number = state_number

    def fit(self, sequences):
        
        self.states = set(x for x in range(self.state_number))

        state_index = {state: i for i, state in enumerate(self.states)}
        n_states = len(self.states)
        
        transition_matrix = np.ones((n_states, n_states))*1e-10

        for sequence in sequences:
            for i in range(len(sequence) - 1):
                state_from, state_to = sequence[i], sequence[i + 1]
                transition_matrix[state_index[state_from]][state_index[state_to]] += 1

        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        self.transition_matrix = transition_matrix

    def predict_sequence_probability(self, sequence):
        state_index = {state: i for i, state in enumerate(self.states)}
        probability = 1.0
        for i in range(len(sequence) - 1):
            probability *= self.transition_matrix[state_index[sequence[i]]][state_index[sequence[i + 1]]]

        return probability


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


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--model', default='OCSVM', type=str, metavar='MODEL',
                        help='Name of model to train: GMM/NB/LocalOutlierFactor/IsolationForest/MC/OCSVM')
    parser.add_argument('--dataset', default='fr', type=str, metavar='MODEL',
                        help='Name of dataset to train: an/fr/us/sp')
    parser.add_argument('--env', default='single', type=str, metavar='type',
                        help='Name of dataset to train: winter/daytime/single')
    parser.add_argument('--singleness_attack', default='multiple_attack_tv', type=str, metavar='type',
                        help='Name of dataset to train: SD/MD/DM/DD')
    return parser


def make_data(args, train_file1, test_file2):
    with open(train_file1, 'rb') as file1:
        X_trn_r1 = pickle.load(file1)

    with open(f"baseline_data/{args.dataset}/attack/{args.dataset}_{args.singleness_attack}.pkl", 'rb') as file3:
        X_test_e = pickle.load(file3)

    with open(test_file2, 'rb') as file2:
        X_test_r = pickle.load(file2)

    return X_trn_r1, X_test_r, X_test_e


def count_cm(labels, y_pred_test, dataset, env, attack_type):
    cm = confusion_matrix(y_true=labels, y_pred=y_pred_test)
    TN, FP, FN, TP = cm.ravel()

    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    recall = recall_score(y_pred=y_pred_test, y_true=labels, zero_division='warn')
    precision = precision_score(y_pred=y_pred_test, y_true=labels, zero_division='warn')
    accuracy = accuracy_score(y_pred=y_pred_test, y_true=labels)
    f1 = f1_score(y_pred=y_pred_test, y_true=labels, zero_division='warn')

    res = {"dataset": dataset, "env": env, "type": attack_type,
           "recall": recall, "precision": precision,  "f1_score": f1}

    return res


def train(args, train_file1, test_file2):
    X_train, X_test_r, X_test_e = make_data(args, train_file1, test_file2)

    labels = [0] * len(X_train) + [0] * len(X_test_r) + [1] * len(X_test_e)
    
    if args.model == "GMM":
        X_test = X_train + X_test_r + X_test_e
        X_test_features = X_test
        
        model = GaussianMixture(n_components=2, random_state=40)
        y_pred_test = model.fit_predict(X_test_features)

    elif args.model == "NB":
        X_test = X_train + X_test_r + X_test_e
        X_test_features = X_test
        labels_NB = [0] * len(X_train)
        
        model = GaussianNB()
        model.fit(X_train, labels_NB)
        y_pred_test = model.predict(X_test_features)

    elif args.model == "LocalOutlierFactor":
        X_test = X_train + X_test_r + X_test_e
        X_test_features = X_test
        
        model = LocalOutlierFactor(n_neighbors=min(20, len(X_test_features)-1), contamination=0.2)
        lof_pred = model.fit_predict(X_test_features)
        y_pred_test = np.where(lof_pred == 1, 0, 1)

    elif args.model == "IsolationForest":
        X_test = X_train + X_test_r + X_test_e
        X_test_features = X_test

        model = IsolationForest(contamination='auto', random_state=42)
        if_pred = model.fit_predict(X_test_features)
        y_pred_test = np.where(if_pred == 1, 0, 1)

    elif args.model == "MC":
        X_test = X_train + X_test_r + X_test_e
        model = MarkovChain(state_number=vocab_dic[args.dataset])
        model.fit(X_train)

        threshold = 1e-50
        y_pred_test = []
        for sequence in X_test:
            probability = model.predict_sequence_probability(sequence)
            
            if probability < threshold:
                y_pred_test.append(0)
            else:
                y_pred_test.append(1)

    else:
        X_test = X_train + X_test_r + X_test_e
        X_test_features = X_test

        model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        ocsvm_pred = model.fit_predict(X_test_features)
        y_pred_test = np.where(ocsvm_pred == 1, 0, 1)

    return count_cm(labels, y_pred_test, args.dataset, args.env, args.singleness_attack)


def train_autoencoder(args, train_file1, test_file2):
    X_train, X_test_r, X_test_e = make_data(args, train_file1, test_file2)
    labels = [0] * len(X_test_r) + [1] * len(X_test_e)

    y_pred_test = anomaly_detection(args, train_file1, test_file2)

    return count_cm(labels, y_pred_test, args.dataset, args.env, args.singleness_attack)


if __name__ == "__main__":
    setup_seed(2024)
    args = get_args_parser()
    args = args.parse_args()

    results = []
    import json
    from datetime import datetime

    models = ["LocalOutlierFactor", "IsolationForest", "MC", "OCSVM", "Autoencoder", "GRUAutoencoder", "TransformerAutoencoder"]
    datasets = ["fr", "sp", "us"]
    environments = ["winter", "daytime", "single"]

    attack_mapping = {
        'winter': 'spring_attack_heater',
        'daytime': 'night_attack_time',
        'single': 'multiple_attack_tv'
    }
    
    print("Start running the anomaly detection baseline experiment ...")

    for env in environments:
        args.env = env
        args.singleness_attack = attack_mapping[env]
        env_results = []
        
        print(f"\n{'='*50} Processing: {env} {'='*50}")

        for model in models:
            args.model = model
            for dataset in datasets:
                args.dataset = dataset

                train_file1 = f"baseline_data/{args.dataset}/{args.env}/trn.pkl"
                vld_file = f"baseline_data/{args.dataset}/{args.env}/vld.pkl"
                test_file2 = f"baseline_data/{args.dataset}/{args.env}/test.pkl"
                
                try:
                    print(f"Run {model} in {dataset} dataset...")
                    if model in ["GMM", "LocalOutlierFactor", "IsolationForest", "MC", "OCSVM"]:
                        res = train(args, train_file1, test_file2)
                    else:
                        res = train_autoencoder(args, train_file1, test_file2)
                    tmp = {model: res}
                    env_results.append(tmp)
                    results.append(tmp)
                    print(f"✓ Finish {model} in {dataset} dataset")
                except Exception as e:
                    print(f"✗ Error processing {model} on {dataset} dataset: {str(e)}")
                    continue

    if results:
        processed_results = []
        for result_dict in results:
            model_name = list(result_dict.keys())[0]
            data = result_dict[model_name]

            data['model'] = model_name

            processed_results.append(data)

        df = pd.DataFrame(processed_results)
        
        columns_order = ['model', 'dataset', 'env', 'type', 'precision', 'recall', 'f1_score']
        df = df[columns_order]
        
        float_cols = ['precision', 'recall', 'f1_score']
        for col in float_cols:
            df[col] = df[col].map(lambda x: f"{x:.4f}")
        
        df = df.sort_values(['env', 'model', 'dataset'])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_result_file = f"results/anomaly_detection_all_{timestamp}.csv"
        print(f"\nSave the total result to: {all_result_file}")
        df.to_csv(all_result_file, index=False)
        
        with open(f"results/anomaly_detection_all_{timestamp}.json", "w") as file_res:
            json.dump(results, file_res, indent=4)
        
        print("\n" + "="*100)
        print("Experiment completed！")
        print("Tag definition: 0=normal sample, 1=attack sample")
        print("TP=True Example, TN=True Negative Example, FP=False Positive Example, FN=False Negative Example")
        print("="*100)
        print("\nThe total result:")
        print(df.to_string(index=False))