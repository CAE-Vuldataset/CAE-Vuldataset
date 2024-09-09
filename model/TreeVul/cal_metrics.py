import numpy as np
from sklearn import metrics
from typing import Any, Dict
import matplotlib.pyplot as plt
import pickle as pkl
from numpy.core.fromnumeric import sort
from pandas.core.frame import DataFrame

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef

import json
from queue import Queue
import copy

RESULT_PATH = "xxx"

def cal_metrics_cwe_ml_baseline(merged_results, depth=0, cwe_path: dict = None, topk=1):
    # for ml baselines
    true_path = []
    predict_path = []
    for meta in merged_results:
        if len(meta["path"]) > depth:
            true_path.append(meta["path"])

            label_prob = meta["prob"]
            label_prob.sort(key=lambda x:x[1], reverse=True)
            topk_predict_path = [cwe_path[_[0]] for _ in label_prob[:topk]]
            max_match = -1
            selected_predict_path = None
            true_path_set = set(meta["path"])
            for path in topk_predict_path:
                match = len(set(path) & true_path_set)
                if match > max_match:
                    max_match = match
                    selected_predict_path = path
            predict_path.append(selected_predict_path)
            
    classif_metrics = path_metric(true_path=true_path, predict_path=predict_path, depth=depth)

    return classif_metrics


def cal_metrics_cwe_tree(file, max_depth=0, cwe_path: dict = None, topk = 1):
    # for neural models that use tree structure information (TreeVul, TreeVul-t)
    merged_results = list()
    f = open(f"{RESULT_PATH}/test_results/{file}.json", 'r')
    for line in f.readlines():
        # outputs of multiple batches is segmented by '\n'
        merged_results.extend(json.loads(line))

    metrics_all = {}
    # for d in range(max_depth):
    for d in [max_depth-1]:
        # we only need the result at max_depth-1
        true_path = []
        predict_path = []
        for meta in merged_results:
            if len(meta["path"]) > d:
                true_path.append(meta["path"])

                # support top k
                topk_predict_path = [cwe_path[meta[f"predict_{d}"][k]] for k in range(topk)]
                max_match = -1
                selected_predict_path = None
                true_path_set = set(meta["path"])
                for path in topk_predict_path:
                    match = len(set(path) & true_path_set)
                    if match > max_match:
                        max_match = match
                        selected_predict_path = path
                predict_path.append(selected_predict_path)
        
        classif_metrics = path_metric(true_path=true_path, predict_path=predict_path, depth=d)
        
        # for sub_metric_name, result in classif_metrics.items():
        #     metrics_all[f"depth-{d}_{sub_metric_name}"] = result
        metrics_all = classif_metrics

    fn = file.split("_")[:-1]
    fn.append(f"metric_all_top{topk}")
    fn = '_'.join(list(fn))
    
    with open(f"{RESULT_PATH}/test_results/{fn}.json", 'w') as f:
        json.dump(metrics_all, f, indent=4)
        

def cal_metrics_cwe(file, depth=0, cwe_path: dict = None, label_idx: list = None, topk = 1):
    # for nueral models that do not use tree structure information (TreeVul-h, nueral baselines)
    merged_results = list()
    f = open(f"{RESULT_PATH}/test_results/{file}.json", 'r')
    for line in f.readlines():
        # outputs of multiple batches is segmented by '\n'
        merged_results.extend(json.loads(line))

    true_path = []
    predict_path = []
    for meta in merged_results:
        if len(meta["path"]) > depth:
            true_path.append(meta["path"])

            # support topk
            label_prob = [(l, p) for l, p in zip(label_idx[depth], meta["prob"])]
            label_prob.sort(key=lambda x:x[1], reverse=True)
            topk_predict_path = [cwe_path[_[0]] for _ in label_prob[:topk]]
            max_match = -1
            selected_predict_path = None
            true_path_set = set(meta["path"])
            for path in topk_predict_path:
                match = len(set(path) & true_path_set)
                if match > max_match:
                    max_match = match
                    selected_predict_path = path
            predict_path.append(selected_predict_path)
            
    classif_metrics = path_metric(true_path=true_path, predict_path=predict_path, depth=depth)

    fn = file.split("_")[:-1]
    fn.append(f"metric_all_top{topk}")
    fn = '_'.join(list(fn))
    
    with open(f"{RESULT_PATH}/test_results/{fn}.json", 'w') as f:
        json.dump(classif_metrics, f, indent=4)


def path_metric(true_path: list, predict_path: list, depth=0, top_k=1):
    print(f"valid test sample for depth-{depth}: {len(true_path)}")

    max_depth = depth + 1
    
    classif_metrics = dict()
    average_modes = ["weighted", "macro"]

    # metrics for each depth
    for d in range(max_depth):
        labels = [_[d] for _ in true_path]
        predicts = [_[d] for _ in predict_path]

        valid_labels = list(set(labels))
        valid_labels.sort()
        for mode in average_modes:
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predicts, average=mode, labels=valid_labels)
            classif_metrics[f"{d}_{mode}_precision"] = precision
            classif_metrics[f"{d}_{mode}_recall"] = recall
            classif_metrics[f"{d}_{mode}_fscore"] = f1
        
        classif_metrics[f"{d}_mcc"] = matthews_corrcoef(labels, predicts)

    # overall metrics
    hierarchy_metric = 0
    for tp, pp in zip(true_path, predict_path):
        hierarchy_metric += len(set(tp[:max_depth]) & set(pp[:max_depth]))
    classif_metrics["overall_hierarchy"] = hierarchy_metric / (max_depth * len(true_path))

    label_all = []
    predict_all = []
    for tp, pp in zip(true_path, predict_path):
        label_all.extend(tp[:max_depth])
        predict_all.extend(pp[:max_depth])
    classif_metrics["overall_accuracy"] = accuracy_score(label_all, predict_all)  # same to metric_hierarchy

    return classif_metrics