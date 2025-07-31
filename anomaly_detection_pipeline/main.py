import argparse
import datetime
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from Anomaly_Detection_pipeline_model import Anomaly_detection

vocab_dic = {"fr": 223, "sp": 235, "us": 269}
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


def process_result(results):
    results_df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/SmartGen_results_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(results_file, index=False)
    results_json = f"results/SmartGen_results_{timestamp}.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved:")
    print(f"CSV: {results_file}")
    print(f"JSON: {results_json}")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--model', default='Transformer', type=str, metavar='MODEL',
                        help='Name of model to train: Transformer')
    parser.add_argument('--dataset', default='fr', type=str, metavar='MODEL',
                        help='Name of dataset to train: an/fr/us/sp')
    parser.add_argument('--env', default='single', type=str, metavar='type',
                        help='Name of dataset to train: winter/daytime/single')
    parser.add_argument('--singleness_attack', default='multiple_attack_tv', type=str, metavar='type',
                        help='Name of dataset to train: SD/MD/DM/DD')
    return parser


if __name__ == "__main__":
    setup_seed(2024)
    start_time = datetime.datetime.now()
    args = get_args_parser()
    args = args.parse_args()

    results = []

    datasets = ["fr", "sp", "us"]
    environments = ["spring", "night", "multiple"]
    thresholds = ['0.918', '0.92', '0.915', '0.915', '0.917', '0.915', '0.905', '0.919', '0.913']
    percentage = [95.5, 95, 99, 95, 95, 99, 95, 93, 99]
    i = 0
    j = 0

    attack_mapping = {
        'spring': 'spring_attack_heater',
        'night': 'night_attack_time',
        'multiple': 'multiple_attack_tv'
    }

    print("Start running the anomaly detection baseline experiment ...")

    for dataset in datasets:
        args.dataset = dataset

        model = args.model
        for env in environments:
            thres = thresholds[i]
            per = percentage[j]
            i += 1
            j += 1
            print(f"\n{'=' * 50} Processing: {dataset}, {env} {'=' * 50}")
            args.env = env
            args.singleness_attack = attack_mapping[env]

            try:
                print(f"Run {model} in {dataset} dataset...")
                test_metrics = Anomaly_detection(args.dataset, args.env, thres, 'SPPC', 'gpt-4o', per)
                result = {
                    'dataset': args.dataset,
                    'env': args.env,
                    'task1': 'anomaly_detection',
                    **test_metrics
                }
                results.append(result)
                print(f"✓ Finish {model} in {dataset} dataset")
            except Exception as e:
                print(f"✗ Error processing {model} on {dataset} dataset: {str(e)}")
                continue

    if results:
        process_result(results)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nEnd time: {end_time}")
    print(f"Total duration: {duration}")
