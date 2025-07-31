import sys
import os
sys.path.append('..')
import torch
import numpy as np
import random
import datetime
import json
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
from DataSet import FixedLengthActionOnlyDataset, collate_fn_fixed
from hmm import DiscreteHMM

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_hmm(train_loader, n_components, n_classes, n_iter=20):
    # Collect all training sequences
    all_sequences = []
    for input_actions, _ in train_loader:
        for seq in input_actions:
            all_sequences.append(seq.numpy())
    # Flatten to 1D
    X = np.concatenate(all_sequences)
    model = DiscreteHMM(n_state=n_components, x_num=n_classes, iter=n_iter)
    model.train(X)
    return model

def predict_hmm(model, input_actions, n_classes):
    # Predict next action
    state_seq = model.decode(input_actions.numpy())
    last_state = int(state_seq[-1])
    emissionprob = model.emission_prob[last_state]
    return emissionprob


def calculate_topk_metrics(outputs, targets, k):
    correct = 0
    total = len(targets)
    top_k_pred = np.argsort(outputs, axis=1)[:, -k:]
    for i in range(total):
        if targets[i] in top_k_pred[i]:
            correct += 1
    return correct / total if total > 0 else 0


def calculate_ndcg_at_k(outputs, targets, k=10):
    outputs = torch.tensor(outputs)
    batch_size = outputs.size(0)
    ndcg_scores = []

    for i in range(batch_size):
        # Get top-k predicted indices and their scores
        scores, pred_indices = torch.topk(outputs[i], k, dim=0)
        # Ground-truth relevance: 1 if the target is in top-k, 0 otherwise
        relevance = np.zeros(k)
        target = targets[i].item()
        for j, pred in enumerate(pred_indices):
            if pred.item() == target:
                relevance[j] = 1  # Binary relevance: 1 for correct, 0 for incorrect

        # Calculate DCG@k
        dcg = 0.0
        for j in range(k):
            if relevance[j] > 0:
                dcg += (2 ** relevance[j] - 1) / np.log2(j + 2)  # j+2 because ranks start at 1

        # Calculate IDCG@k (ideal DCG, where the relevant item is ranked first)
        idcg = (2 ** 1 - 1) / np.log2(2)

        # NDCG@k = DCG@k / IDCG@k
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0

def evaluate_model(model, test_loader, n_classes):
    all_targets = []
    all_outputs = []
    all_predictions = []
    for input_actions, targets in tqdm(test_loader, desc="Evaluating"):
        for seq, target in zip(input_actions, targets):
            emissionprob = predict_hmm(model, seq, n_classes)
            all_outputs.append(emissionprob)
            pred = np.argmax(emissionprob)
            all_predictions.append(pred)
            all_targets.append(target.item())
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    acc_1 = calculate_topk_metrics(all_outputs, all_targets, 1)
    acc_3 = calculate_topk_metrics(all_outputs, all_targets, 3)
    acc_5 = calculate_topk_metrics(all_outputs, all_targets, 5)
    acc_10 = calculate_topk_metrics(all_outputs, all_targets, 10)
    ndcg = calculate_ndcg_at_k(all_outputs, all_targets, 10)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    return {
        'ACC@1': acc_1,
        'ACC@3': acc_3,
        'ACC@5': acc_5,
        'HR@10': acc_10,
        'NDCG@10': ndcg,
        'Macro-F1': macro_f1
    }

def run_experiment(dataset, environment, args):
    print(f"\n{'='*50}")
    print(f"Start experiment: {dataset}/{environment}")
    print(f"{'='*50}")
    data_dir = f"../../baseline_data/{dataset}/{environment}"
    train_path = os.path.join(data_dir, "trn.pkl")
    val_path = os.path.join(data_dir, "vld.pkl")
    test_path = os.path.join(data_dir, "test.pkl")
    if not os.path.exists(train_path):
        print(f"Error: Training file does not exist {train_path}")
        return None
    if not os.path.exists(val_path):
        print(f"Error: Validation file does not exist {val_path}")
        return None
    if not os.path.exists(test_path):
        print(f"Error: Test file does not exist {test_path}")
        return None
    train_dataset = FixedLengthActionOnlyDataset(train_path)
    val_dataset = FixedLengthActionOnlyDataset(val_path)
    test_dataset = FixedLengthActionOnlyDataset(test_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_fixed)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_fixed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_fixed)
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    # n_classes is the max action in train+val+test plus 1
    n_classes = max([
        max(seq) for seq in train_dataset.data + val_dataset.data + test_dataset.data
    ]) + 1
    n_components = min(args.n_components, n_classes)
    print(f"HMM number of hidden states: {n_components}, number of action classes: {n_classes}")
    hmm_model = train_hmm(train_loader, n_components=n_components, n_classes=n_classes, n_iter=args.n_iter)
    test_metrics = evaluate_model(hmm_model, test_loader, n_classes)
    result = {
        'dataset': dataset,
        'environment': environment,
        'model': 'HMM_fixed',
        **test_metrics
    }
    return result

def main():
    parser = argparse.ArgumentParser(description='HMM Fixed Length Action-Only Training')
    parser.add_argument('--n_components', type=int, default=30, help='Number of HMM hidden states')
    parser.add_argument('--n_iter', type=int, default=30, help='Max HMM iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--datasets', nargs='+', default=['fr', 'sp', 'us'], help='Datasets to process')
    parser.add_argument('--environments', nargs='+', default=['winter', 'daytime', 'single'], help='Environments to process')
    args = parser.parse_args()
    setup_seed(args.seed)
    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    print(f"Start time: {start_time}")
    print(f"Args: {vars(args)}")
    results = []
    for dataset in args.datasets:
        for environment in args.environments:
            try:
                result = run_experiment(dataset, environment, args)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Experiment {dataset}/{environment} failed: {e}")
                continue
    if results:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"action_only_results_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        results_json = os.path.join(results_dir, f"action_only_results_{timestamp}.json")
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved:")
        print(f"CSV: {results_file}")
        print(f"JSON: {results_json}")
        print(f"\n{'='*50}")
        print("Experiment results summary:")
        print(f"{'='*50}")
        print(results_df.to_string(index=False))
        print(f"\nAverage performance:")
        for metric in ['ACC@1', 'ACC@3', 'ACC@5', 'HR@10', 'NDCG@10', 'Macro-F1']:
            avg_value = results_df[metric].mean()
            print(f"  {metric}: {avg_value:.4f}")
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nEnd time: {end_time}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main() 