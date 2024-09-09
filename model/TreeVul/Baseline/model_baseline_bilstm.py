from json import encoder
import logging
from typing import Dict, List, Any

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.common import Params
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder
from allennlp.modules.seq2seq_encoders.pytorch_transformer_wrapper import PytorchTransformer
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder, BertPooler, ClsPooler

from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric, metric
from allennlp.training.util import get_batch_size

from torch import embedding, nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable

from .custom_metric import ClassifPathMetric
from .custom_modules import pooling_with_mask

import warnings
import json
from copy import deepcopy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@Model.register("model_baseline_bilstm")
class ModelBaselineBiLSTM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.1,
                 label_file: str = 'valid_cwes.json',
                 cwe_path_file: str = 'cwe_path.json',
                 invalid_label_index: int = -1,
                 depth: int = 0,
                 device: str = "cpu",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab)
        self._device = torch.device(device)
        self._dropout = Dropout(dropout)
        
        self._labels = json.load(open(label_file, 'r'))
        self._invalid_label_index = invalid_label_index
        self._depth = depth
        
        self._text_field_embedder = text_field_embedder
        embedding_dim = self._text_field_embedder.get_output_dim()
        self._encoder = LstmSeq2SeqEncoder(
                        input_size=embedding_dim,
                        hidden_size=int(embedding_dim / 2),
                        num_layers=1,
                        bias=True,
                        dropout=0.1,
                        bidirectional=True)
        embedding_dim = self._encoder.get_output_dim()
        
        hidden_dim = 128
        self._projector = nn.Sequential(
            FeedForward(embedding_dim, 1, [hidden_dim], torch.nn.ReLU(), dropout),
            nn.Linear(hidden_dim, len(self._labels[self._depth])),
        )

        # only focus on custom metrics
        cwe_path = json.load(open(cwe_path_file, 'r'))
        self._custom_metrics = ClassifPathMetric(depth=depth, cwe_path=cwe_path)

        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1-score_weighted": FBetaMeasure(beta=1.0, average="weighted", labels=range(len(self._labels[self._depth]))),  # return float
        }

        self._loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=self._invalid_label_index, reduction="mean")
        initializer(self)

    def forward(self,
                diff: TextFieldTensors,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:

        output_dict = dict()
        if metadata:
            output_dict["meta"] = metadata

        diff_mask = get_text_field_mask(diff, num_wrapping_dims=1, padding_id=0)
        diff_hunk_mask = diff_mask.float().sum(dim=-1) > 0
       
        # sequence field
        diff_embedding = self._text_field_embedder(diff, num_wrapping_dims=1)
        shape = diff_embedding.shape

        diff_embedding = diff_embedding.view(-1, shape[-2], shape[-1])
        diff_embedding = self._encoder(inputs=diff_embedding, mask=diff_mask.view(-1, shape[-2]))

        diff_embedding_pooled = pooling_with_mask(input=diff_embedding, mask=diff_mask.view(-1, shape[-2]))
        diff_embedding_pooled = pooling_with_mask(input=diff_embedding_pooled.view(shape[0], shape[1], -1), mask=diff_hunk_mask)

        logits = self._projector(diff_embedding_pooled)
        loss = self._loss(logits, label)
        output_dict['loss'] = loss

        probs = nn.functional.softmax(logits, dim=-1)
        output_dict["probs"] = probs

        predicts = torch.argmax(probs, dim=-1)
        output_dict["predicts"] = predicts
        self._custom_metrics(predictions=[self._labels[self._depth][idx] for idx in predicts.tolist()], metadata=metadata)
        
        for metric_name, metric_ in self._metrics.items():
            if metadata[0]["type"] != "test" or "f1-score" not in metric_name:
                # in test, it is possible to have new labels (index=-1), metric API for f1-score will fail
                metric_(predictions=probs, gold_labels=label)

        return output_dict
    
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        label_idx = output_dict["predicts"].tolist()
        probs = output_dict["probs"].tolist()
        out2file = list()
        for i, idx in enumerate(label_idx):
            meta = output_dict["meta"][i]["instance"]
            meta["predict"] = self._labels[self._depth][idx]
            meta["prob"] = probs[i]
            out2file.append(meta)
        return out2file

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        
        metrics['accuracy'] = self._metrics['accuracy'].get_metric(reset)
        try:
            precision, recall, fscore = self._metrics['f1-score_weighted'].get_metric(reset).values()
            metrics['weighted_precision'] = precision
            metrics['weighted_recall'] = recall
            metrics['weighted_f1-score'] = fscore
        except Exception:
            pass
        
        custom_metrics = deepcopy(self._custom_metrics.get_metric(reset))

        metrics = dict(metrics, **custom_metrics)

        return metrics