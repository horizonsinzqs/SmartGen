import sys
import os
sys.path.append('..')  # Add parent directory to path

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
import CARNN_fixed
import DataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Dataset config
DATASET_CONFIG = {
    'fr': {'action_num': 222, 'device_num': 33},
    'sp': {'action_num': 234, 'device_num': 34},
    'us': {'action_num': 268, 'device_num': 40}
}

ENVIRONMENTS = ['daytime', 'single', 'winter']

def setup_seed(seed):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def create_model(dataset, d_model=64, max_sequence_length=50, timeStamp=7*8):
    """Create model"""
    config = DATASET_CONFIG[dataset]
    model = CARNN_fixed.CARNN_fixed(
        input_length=9,  # First 9 actions
        d_model=d_model,
        actionNum=config['action_num'],
        max_sequence_length=max_sequence_length,
        timeStamp=timeStamp
    )
    return model

def train_model(model, train_loader, val_loader, args, save_dir):
    """Train model"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Enable GPU optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    print(f"Start training, total {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in train_bar:
            input_actions, targets = batch
            input_actions = input_actions.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_actions)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        lr_scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_actions, targets = batch
                input_actions = input_actions.to(device)
                targets = targets.to(device)
                
                outputs = model(input_actions)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * train_correct / train_total
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  Save best model, validation accuracy: {val_acc:.2f}%")
    
    return best_val_acc

def calculate_topk_metrics(outputs, targets, k):
    """Calculate Top-k accuracy"""
    correct = 0
    total = len(targets)
    _, top_k_pred = torch.topk(outputs, k, dim=1)
    for i in range(total):
        if targets[i] in top_k_pred[i]:
            correct += 1
    return correct / total if total > 0 else 0


def calculate_ndcg_at_k(outputs, targets, k=10):
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

def evaluate_model(model, test_loader, model_path):
    """Evaluate model, return multiple metrics"""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_actions, targets = batch
            input_actions = input_actions.to(device)
            targets = targets.to(device)
            outputs = model(input_actions)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Top-k accuracy
    acc_1 = calculate_topk_metrics(all_outputs, all_targets, 1)
    acc_3 = calculate_topk_metrics(all_outputs, all_targets, 3)
    acc_5 = calculate_topk_metrics(all_outputs, all_targets, 5)
    acc_10 = calculate_topk_metrics(all_outputs, all_targets, 10)
    ndcg = calculate_ndcg_at_k(all_outputs, all_targets, 10)
    # Macro-F1
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
    """Run a single experiment"""
    print(f"\n{'='*50}")
    print(f"Start experiment: {dataset}/{environment}")
    print(f"{'='*50}")
    
    # Data path
    data_dir = f"../../baseline_data/{dataset}/{environment}"
    train_path = os.path.join(data_dir, "trn.pkl")
    val_path = os.path.join(data_dir, "vld.pkl")
    test_path = os.path.join(data_dir, "test.pkl")
    
    # Check if files exist
    if not os.path.exists(train_path):
        print(f"Error: Training file does not exist {train_path}")
        return None
    if not os.path.exists(val_path):
        print(f"Error: Validation file does not exist {val_path}")
        return None
    if not os.path.exists(test_path):
        print(f"Error: Test file does not exist {test_path}")
        return None
    
    # Create data loaders
    train_dataset = DataSet.FixedLengthDataSet(train_path)
    val_dataset = DataSet.FixedLengthDataSet(val_path)
    test_dataset = DataSet.FixedLengthDataSet(test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=DataSet.collate_fn_fixed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=DataSet.collate_fn_fixed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=DataSet.collate_fn_fixed)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create model
    model = create_model(dataset, args.d_model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save directory
    save_dir = f"save_weights_{dataset}_{environment}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Train model
    best_val_acc = train_model(model, train_loader, val_loader, args, save_dir)
    
    # Evaluate model
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    test_metrics = evaluate_model(model, test_loader, best_model_path)
    
    result = {
        'dataset': dataset,
        'environment': environment,
        'model': 'CARNN_fixed',
        **test_metrics
    }
    return result

def main():
    parser = argparse.ArgumentParser(description='CARNN Fixed Length Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=20, help='Learning rate step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--datasets', nargs='+', default=['fr', 'sp', 'us'], help='Datasets to process')
    parser.add_argument('--environments', nargs='+', default=['daytime', 'single', 'winter'], help='Environments to process')
    
    args = parser.parse_args()
    
    # Set random seed
    setup_seed(args.seed)
    
    # Record start time
    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print(f"Start time: {start_time}")
    print(f"Args: {vars(args)}")
    
    # Store results
    results = []
    
    # Run all experiments
    for dataset in args.datasets:
        for environment in args.environments:
            try:
                result = run_experiment(dataset, environment, args)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Experiment {dataset}/{environment} failed: {e}")
                continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/action_only_results_{timestamp}.csv"
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(results_file, index=False)
        
        # Save as JSON
        results_json = f"results/action_only_results_{timestamp}.json"
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved:")
        print(f"CSV: {results_file}")
        print(f"JSON: {results_json}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("Experiment results summary:")
        print(f"{'='*50}")
        print(results_df.to_string(index=False))
        # Calculate average performance
        print(f"\nAverage performance:")
        for metric in ['ACC@1', 'ACC@3', 'ACC@5', 'HR@10', 'NDCG@10', 'Macro-F1']:
            avg_value = results_df[metric].mean()
            print(f"  {metric}: {avg_value:.4f}")
    
    # Record end time
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nEnd time: {end_time}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main() 