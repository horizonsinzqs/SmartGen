import os
import time
import torch
import argparse
import pandas as pd
import datetime
import json

from model import SASRec
from utils import *
from SASRec import SASRec_behavior_prediction, SASRec_behavior_prediction_ab1
from baseline1 import Anomaly_detection, Anomaly_detection_ab1
from security_check import security_check

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


def process_result_ab2(results, study):
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

if __name__ == '__main__':
    ''' Please choose an experiment: ablation1, ablation2 '''
    setup_seed(2024)
    start_time = datetime.datetime.now()
    study = 'ablation2'
    # for study in experiments:
    if study == 'ablation1':
        model = 'Llama_70B'
        threshold = '0.905'
        percentage = 95
        dataset = 'sp'
        environments = "spring"
        modules = ['without_all', 'without_SSC', 'without_TSS', 'without_GSS', 'all']
        results = []
        for wo in modules:
            test_metrics1 = Anomaly_detection_ab1(dataset, environments, threshold, model, study, percentage, wo)
            # SASRec_behavior_prediction_ab1(dataset, environments, threshold, model, study, 'train', wo)
            test_metrics2 = SASRec_behavior_prediction_ab1(dataset, environments, threshold, model, study, 'test', wo)
            result = {
                'module': wo,
                'LLM type': model,
                'task1': 'anomaly_detection',
                **test_metrics1,
                'task2': 'behavior_prediction',
                **test_metrics2
            }
            results.append(result)
        if results:
            process_result(results, study)

    elif study == 'ablation2':
        model = 'gpt-4o'
        thres = ['0.918', '0.92', '0.915', '0.915', '0.917', '0.915', '0.905', '0.919', '0.913']
        percentage = [95.5, 95, 99, 95, 95, 99, 95, 93, 99]
        datasets = ["fr", "sp", "us"]
        environments = ['spring', 'night', 'multiple']
        results = []
        i = 0
        j = 0
        for nation in datasets:
            for env in environments:
                threshold = thres[i]
                per = percentage[j]
                i += 1
                j += 1
                security_check(nation, env, threshold, 'SPPC', model)
                test_metrics1 = Anomaly_detection(nation, env, threshold, "SPPC", model, study, per, 'ori')
                test_metrics2 = Anomaly_detection(nation, env, threshold, "SPPC", model, study, per, 'S1')
                test_metrics3 = Anomaly_detection(nation, env, threshold, "SPPC", model, study, per, 'S2')

                result = {
                    'dataset': nation,
                    'env': env,
                    **test_metrics1,
                    **test_metrics2,
                    **test_metrics3
                }
                results.append(result)
        if results:
            process_result_ab2(results, study)





