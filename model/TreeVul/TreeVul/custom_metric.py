from typing import Optional, List, Dict, Any

from allennlp.training.metrics import Metric, metric
from overrides import overrides
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef


@Metric.register("classif_path_metric")
class ClassifPathMetric(Metric):
    def __init__(self, depth: int = -1, cwe_path: dict = None) -> None:
        self._label_path = list()
        self._predict_path = list()

        self._max_depth = depth + 1
        self._cwe_path = cwe_path
         
    @overrides
    def __call__(self,
                 predictions: list,
                 metadata: List[Dict[str, Any]] = None):
        
        if metadata[0]["type"] == "train":
            return
        
        # we use label (str) for calculating the metrics, instead of label idx (int)
        for meta, cwe in zip(metadata, predictions):
            if len(meta["instance"]["path"]) >= self._max_depth:
                # remove invalid samples
                self._label_path.append(meta["instance"]["path"])
                self._predict_path.append(self._cwe_path[cwe])

    def get_metric(self, reset: bool):
        metrics_ = dict()
        if len(self._label_path) == 0:
            # train, not validation
            return metrics_
        
        average_modes = ["weighted", "macro"]
        
        # also metrics for upper levels
        for depth in range(self._max_depth):
            labels = [label_path[depth] for label_path in self._label_path]
            predicts = [predict_path[depth] for predict_path in self._predict_path]
            
            valid_labels = list(set(labels))
            valid_labels.sort()  # same order every time
            for mode in average_modes:
                precision, recall, f1, _ = precision_recall_fscore_support(labels, predicts, average=mode, labels=valid_labels)
                metrics_[f"{depth}_{mode}_precision"] = precision
                metrics_[f"{depth}_{mode}_recall"] = recall
                metrics_[f"{depth}_{mode}_fscore"] = f1

            # between -1 and 1, 0 equals to random guess
            metrics_[f"{depth}_mcc"] = matthews_corrcoef(labels, predicts)
        
        # overall metric
        hierarchy_metric = 0
        for label_path, predict_path in zip(self._label_path, self._predict_path):
            hierarchy_metric += len(set(label_path[:self._max_depth]) & set(predict_path[:self._max_depth]))

        metrics_["overall_hierarchy"] = hierarchy_metric / (self._max_depth * len(self._label_path))

        # label_all = []
        # predict_all = []
        # for tp, pp in zip(self._label_path, self._predict_path):
        #     label_all.extend(tp[:self._max_depth])
        #     predict_all.extend(pp[:self._max_depth])
        # metrics_["overall_accuracy"] = accuracy_score(label_all, predict_all)  # same to metric_hierarchy

        if reset:
            self.reset()
        
        return metrics_

    def reset(self) -> None:
        self._label_path.clear()
        self._predict_path.clear()