import os
import time
import torch
import argparse
import pandas as pd
import datetime
import json

from model import SASRec
from utils import *
from SASRec import SASRec_behavior_prediction, SASRec_behavior_prediction_splitEP
from baseline1 import Anomaly_detection, Anomaly_detection_splitEP

vocab_dic = {"an": 141, "fr": 222, "us": 268, "sp": 234}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def process_result(results, study):
    results_df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/{study}_results_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(results_file, index=False)
    results_json = f"results/{study}_results_{timestamp}.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved:")
    print(f"CSV: {results_file}")
    print(f"JSON: {results_json}")
    print(f"\n{'=' * 50}")
    print("Experiment results summary:")
    print(f"{'=' * 50}")
    print(results_df.to_string(index=False))
    print(f"\nAverage performance:")
    for metric in ['HR@10', 'NDCG@10']:
        avg_value = results_df[metric].mean()
        print(f"  {metric}: {avg_value:.4f}")

if __name__ == '__main__':
    '''
    Please choose an experiment: 'diffllmEP', 'thresholdEP', splitEP1, splitEP2
    
    '''
    setup_seed(2024)
    start_time = datetime.datetime.now()

    study = 'thresholdEP'
    # for study in experiments:
    if study == 'diffllmEP':
        models = ['Llama_70B', 'Qwen2.5_72B', 'gpt-4o']
        thres = ['0.918', '0.915', '0.905']
        percentage = [95, 93, 95.5, 95, 75, 95, 95, 40, 95]
        datasets = ["fr", "sp", "us"]
        environments = "spring"
        results = []
        i = 0
        j = 0
        for nation in datasets:
            threshold = thres[i]
            i += 1
            for model in models:
                per = percentage[j]
                j += 1
                test_metrics1 = Anomaly_detection(nation, environments, threshold, "SPPC", model, study, per)
                # SASRec_behavior_prediction(nation, environments, threshold, "SPPC", model, study, 'train')
                test_metrics2 = SASRec_behavior_prediction(nation, environments, threshold, "SPPC", model, study, 'test')
                result = {
                    'dataset': nation,
                    'LLM type': model,
                    'task1': 'anomaly_detection',
                    **test_metrics1,
                    'task2': 'behavior_prediction',
                    **test_metrics2
                }
                results.append(result)
        if results:
            process_result(results, study)

    elif study == 'thresholdEP':
        model = 'gpt-4o'
        thres = ['0.918', '0.919', '0.92']
        percentage = [95, 95, 95, 95, 80, 95, 80, 80, 80]
        datasets = ["fr", "sp", "us"]
        environments = "night"
        results = []
        j = 0
        for nation in datasets:
            for threshold in thres:
                per = percentage[j]
                j += 1
                test_metrics1 = Anomaly_detection(nation, environments, threshold, "SPPC", model, study, per)
                # SASRec_behavior_prediction(nation, environments, threshold, "SPPC", model, study, 'train')
                test_metrics2 = SASRec_behavior_prediction(nation, environments, threshold, "SPPC", model, study,
                                                          'test')
                result = {
                    'dataset': nation,
                    'threshold': threshold,
                    'task1': 'anomaly_detection',
                    **test_metrics1,
                    'task2': 'behavior_prediction',
                    **test_metrics2
                }
                results.append(result)
        if results:
            process_result(results, study)

    elif study == 'splitEP1':
        model = 'gpt-4o'
        thres = ['0.918', '0.915']
        percentage = [95, 95.5, 95, 90, 95, 95.5, 95, 90]
        datasets = ["fr", "sp"]
        environments = "spring"
        results = []
        i = 0
        j = 0
        for nation in datasets:
            threshold = thres[i]
            i += 1
            split_gap2 = '24'
            for split_gap1 in ['6', '9', '12', '15']:
                per = percentage[j]
                j += 1
                test_metrics1 = Anomaly_detection_splitEP(nation, environments, threshold, "SPPC", model, study, per, split_gap1, split_gap2)
                # SASRec_behavior_prediction_splitEP(nation, environments, threshold, "SPPC", model, study, split_gap1, split_gap2, 'train')
                test_metrics2 = SASRec_behavior_prediction_splitEP(nation, environments, threshold, "SPPC", model, study, split_gap1, split_gap2,
                                                           'test')
                result = {
                    'dataset': nation,
                    'split gap 1': split_gap1,
                    'task1': 'anomaly_detection',
                    **test_metrics1,
                    'task2': 'behavior_prediction',
                    **test_metrics2
                }
                results.append(result)
        if results:
            process_result(results, study)

    elif study == 'splitEP2':
        model = 'gpt-4o'
        thres = ['0.918', '0.915']
        percentage = [95, 95.5, 95, 95, 95, 95, 95, 95]
        datasets = ["fr", "sp"]
        environments = "spring"
        results = []
        i = 0
        j = 0
        for nation in datasets:
            threshold = thres[i]
            i += 1
            split_gap1 = '9'
            for split_gap2 in ['12', '24', '36', '48']:
                per = percentage[j]
                j += 1
                test_metrics1 = Anomaly_detection_splitEP(nation, environments, threshold, "SPPC", model, study, per, split_gap1, split_gap2)
                # SASRec_behavior_prediction_splitEP(nation, environments, threshold, "SPPC", model, study, split_gap1, split_gap2, 'train')
                test_metrics2 = SASRec_behavior_prediction_splitEP(nation, environments, threshold, "SPPC", model, study, split_gap1, split_gap2,'test')
                result = {
                    'dataset': nation,
                    'split gap 2': split_gap2,
                    'task1': 'anomaly_detection',
                    **test_metrics1,
                    'task2': 'behavior_prediction',
                    **test_metrics2
                }
                results.append(result)

        if results:
            process_result(results, study)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nEnd time: {end_time}")
    print(f"Total duration: {duration}")



