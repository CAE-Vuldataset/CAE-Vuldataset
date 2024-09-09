from cProfile import label
from cgi import test
import csv
import pandas as pd
import numpy as np
import time
import warnings
import json
import random

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from cal_metrics import cal_metrics_cwe_ml_baseline

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

from reader_baseline_ml import reader_ml_baselines

warnings.filterwarnings('ignore')

def selectFromLinearSVC2(train_content, train_label, test_content):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1').fit(train_content, train_label)
    model = SelectFromModel(lsvc, prefit=True)
    
    new_train = model.transform(train_content)
    new_test = model.transform(test_content)
 
    return new_train, new_test


def main():
    '''
    machine baselines use BoW
    '''

    train_data = json.load(open(DATA_PATH + "train_set.json", 'r'))
    validation_data = json.load(open(DATA_PATH + "validation_set.json"))
    train_data = train_data + validation_data

    test_data = json.load(open(DATA_PATH + "test_set.json", 'r'))
    cwe_path = json.load(open(DATA_PATH + "cwe_path.json", 'r'))  # cwe path

    target_depth = 2
    max_hunk_num = 8
    max_length = 128
    top_k = 1

    random.shuffle(train_data)

    train_corpus, train_label_path = reader_ml_baselines(samples_file=train_data,
                                                    max_hunk_num=max_hunk_num,
                                                    target_depth=target_depth,
                                                    tokenizer_max_length=max_length)
    
    test_corpus, test_label_path = reader_ml_baselines(samples_file=test_data,
                                                  max_hunk_num=max_hunk_num,
                                                  target_depth=target_depth,
                                                  tokenizer_max_length=max_length)
    
    train_labels = [path[target_depth] for path in train_label_path]
    test_labels = [path[target_depth] for path in test_label_path]

    print("converting text into matrix ...")
    vectorizer = CountVectorizer(lowercase=True,
                                 stop_words=None,
                                 tokenizer=None,
                                 max_features=10000,
                                 vocabulary=None)

    train_content_matrix = vectorizer.fit_transform(train_corpus)
    test_content_matrix = vectorizer.transform(test_corpus)
    print("feature selection ...")
    train_content_matrix, test_content_matrix = selectFromLinearSVC2(train_content_matrix, train_labels, test_content_matrix)

    train_x = train_content_matrix.toarray()
    train_y = train_labels
    test_x = test_content_matrix.toarray()
    test_y = test_labels

    learners = ["RF", "SVM", "LR", "KNN", "XGB"]
    
    workers = 16
    
    for l in learners:
        if l == "RF":
            print("===============RF training===============")
            clf = RandomForestClassifier(max_depth=None, n_jobs=workers, random_state=SEED)

            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            merged_results = []
            for path, pp in zip(test_label_path, proba):
                label_prob = [(l, p) for l, p in zip(clf.classes_, pp)]
                merged_results.append({"path": path, "prob": label_prob})

            classif_metrics = cal_metrics_cwe_ml_baseline(merged_results, depth=target_depth, cwe_path=cwe_path, topk=top_k)

            print(classif_metrics)
            with open(RESULT_PATH + f"{l}_metric.json", 'w') as f:
                json.dump(classif_metrics, f, indent=4)
            
            with open(RESULT_PATH + f"{l}_result.json", 'w') as f:
                json.dump(merged_results, f, indent=4)
        
        elif l == "SVM":
            print("===============SVM training===============")

            clf = SVC(kernel='rbf', max_iter=-1, probability=True, random_state=SEED)

            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            merged_results = []
            for path, pp in zip(test_label_path, proba):
                label_prob = [(l, p) for l, p in zip(clf.classes_, pp)]
                merged_results.append({"path": path, "prob": label_prob})

            classif_metrics = cal_metrics_cwe_ml_baseline(merged_results, depth=target_depth, cwe_path=cwe_path, topk=top_k)

            print(classif_metrics)
            with open(RESULT_PATH + f"{l}_metric.json", 'w') as f:
                json.dump(classif_metrics, f, indent=4)
            
            with open(RESULT_PATH + f"{l}_result.json", 'w') as f:
                json.dump(merged_results, f, indent=4)
          
        elif l == "LR":
            print("===============LR training===============")
            
            clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', tol=0.001, max_iter=1000, n_jobs=workers, random_state=SEED)

            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            merged_results = []
            for path, pp in zip(test_label_path, proba):
                label_prob = [(l, p) for l, p in zip(clf.classes_, pp)]
                merged_results.append({"path": path, "prob": label_prob})

            classif_metrics = cal_metrics_cwe_ml_baseline(merged_results, depth=target_depth, cwe_path=cwe_path, topk=top_k)

            print(classif_metrics)
            with open(RESULT_PATH + f"{l}_metric.json", 'w') as f:
                json.dump(classif_metrics, f, indent=4)
            
            with open(RESULT_PATH + f"{l}_result.json", 'w') as f:
                json.dump(merged_results, f, indent=4)
        
        elif l == "KNN":
            print("===============KNN training===============")
            clf = KNeighborsClassifier(n_jobs=workers)

            clf.fit(train_x, train_y)
            proba = clf.predict_proba(test_x)

            merged_results = []
            for path, pp in zip(test_label_path, proba):
                label_prob = [(l, p) for l, p in zip(clf.classes_, pp)]
                merged_results.append({"path": path, "prob": label_prob})

            classif_metrics = cal_metrics_cwe_ml_baseline(merged_results, depth=target_depth, cwe_path=cwe_path, topk=top_k)

            print(classif_metrics)
            with open(RESULT_PATH + f"{l}_metric.json", 'w') as f:
                json.dump(classif_metrics, f, indent=4)
            
            with open(RESULT_PATH + f"{l}_result.json", 'w') as f:
                json.dump(merged_results, f, indent=4)    

        elif l == "XGB":
            print("===============XGB training===============")

            # we need to transfer labels into indexes for XGB
            classes = sorted(list(set(train_y)))
            train_y_index = [classes.index(y) for y in train_y]

            clf = XGBClassifier(objective='multi:softprob', n_jobs=workers, random_state=SEED)

            clf.fit(train_x, train_y_index)
            proba = clf.predict_proba(test_x)

            merged_results = []
            for path, pp in zip(test_label_path, proba):
                label_prob = [(classes[l], float(p)) for l, p in zip(clf.classes_, pp)]
                merged_results.append({"path": path, "prob": label_prob})

            classif_metrics = cal_metrics_cwe_ml_baseline(merged_results, depth=target_depth, cwe_path=cwe_path, topk=top_k)

            print(classif_metrics)
            with open(RESULT_PATH + f"{l}_metric.json", 'w') as f:
                json.dump(classif_metrics, f, indent=4)
            
            with open(RESULT_PATH + f"{l}_result.json", 'w') as f:
                json.dump(merged_results, f, indent=4)


SEED = 2022
DATA_PATH = "xxx"
RESULT_PATH = "xxx"

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    main()