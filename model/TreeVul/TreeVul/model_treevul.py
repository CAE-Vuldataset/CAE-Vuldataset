import logging
from typing import Dict, List, Any, Union

from overrides import overrides
from pydantic import NoneIsNotAllowedError

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.common import Params
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder, BertPooler, ClsPooler
from allennlp.modules.seq2seq_encoders.pytorch_transformer_wrapper import PytorchTransformer
from allennlp.modules.gated_sum import GatedSum

from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric, metric
from allennlp.training.util import get_batch_size

from torch import embedding, logit, nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import random

from .custom_metric import ClassifPathMetric
from .custom_modules import pooling_with_mask

import warnings
import json
from copy import deepcopy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@Model.register("model_treevul")
class ModelTreeVul(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 PTM: str = 'microsoft/codebert-base',
                 dropout: float = 0.1,
                 label_file: str = 'valid_cwes.json',
                 cwe_path_file: str = 'cwe_path.json',
                 invalid_label_index: int = -1,
                 max_depth: int = 3,
                 depth_weights: list = None,
                 device: str = "cpu",
                 train_label_embedding: bool = True,
                 beam_size: int = 3,
                 top_k: int = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab)
        self._device = torch.device(device)
        self._dropout = Dropout(dropout)
        
        self._labels = json.load(open(label_file, 'r'))
        self._invalid_label_index = invalid_label_index
        self._max_depth = max_depth  # only for training

        self._train_label_embedding = train_label_embedding
        self._pooler = BertPooler(PTM, requires_grad=True, dropout=dropout)  # for label embedding
        self._teacher_forcing_ratio = None  # set by callbacks

        # shared backbone
        self._text_field_embedder = text_field_embedder
        embedding_dim = self._text_field_embedder.get_output_dim()
        hidden_dim_bilstm = int(embedding_dim / 2)

        # task-specific headers
        for depth in range(max_depth):
            if depth != 0:
                # support gated sum of diff embedding and label embedding
                setattr(self, f"_{depth}_merger",
                        GatedSum(input_dim=embedding_dim,
                                 activation=torch.nn.Sigmoid()))  # activation function should generate value in [0,1]

                # depth-specific encoder
                setattr(self, f"_{depth}_encoder",
                        LstmSeq2SeqEncoder(
                            input_size=embedding_dim,
                            hidden_size=hidden_dim_bilstm,
                            num_layers=1,
                            bias=True,
                            dropout=0.1,
                            bidirectional=True))

            hidden_dim = 512
            setattr(self, f"_{depth}_projector",
                    nn.Sequential(
                        FeedForward(embedding_dim, 1, [hidden_dim], torch.nn.ReLU(), dropout),
                        nn.Linear(hidden_dim, len(self._labels[depth]))
                    ))
        
        # only focus on custom metrics
        cwe_path = json.load(open(cwe_path_file, 'r'))
        self._custom_metrics = ClassifPathMetric(depth=(self._max_depth-1), cwe_path=cwe_path)
        
        self._metrics = {
            "depth-0_accuracy": CategoricalAccuracy(),
            "depth-0_fscore_weighted": FBetaMeasure(beta=1.0, average="weighted", labels=range(len(self._labels[0])))
        }

        self._loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=self._invalid_label_index, reduction="mean")
        self.set_loss_weights(depth_weights)

        # only for inference
        self._beam_size = beam_size
        self._top_k = top_k

        initializer(self)
    
    def set_loss_weights(self, depth_weights):
        if len(depth_weights) != self._max_depth:
            raise ValueError("length of depth_weights should equal to max_depth")
        self._depth_weights = torch.tensor(depth_weights, device=self._device, dtype=torch.float)
        self._depth_weights /= sum(depth_weights)
    
    def forward_cwe_description(self,
                                description: TextFieldTensors = None,
                                metadata: List[Dict[str, Any]] = None):
        '''
        if we don't need the chain model, then we can deletet this function
        '''

        if description is None:
            # during training
            # self._cwe_tree has already been initilized (contain token_ids), need to update the corresponding embedding
            total_cwes = list(self._cwe_tree.keys())
            description = {"tokens": {"token_ids": torch.stack([self._cwe_tree[cwe]["description"]["tokens"]["token_ids"] for cwe in total_cwes]),
                                      "mask": torch.stack([self._cwe_tree[cwe]["description"]["tokens"]["mask"] for cwe in total_cwes]),
                                      "type_ids": torch.stack([self._cwe_tree[cwe]["description"]["tokens"]["type_ids"] for cwe in total_cwes])}}
            desciprtion_embed = self._text_field_embedder(description)
            desciprtion_embed = self._pooler(desciprtion_embed)
            for index, cwe in enumerate(total_cwes):
                self._cwe_tree[cwe]["description_embed"] = desciprtion_embed[index].unsqueeze(0)
            return

        # initialize the cwe tree
        self._cwe_tree = dict()
        if self._train_label_embedding:
            # for training
            for index, meta in enumerate(metadata):
                des = {"tokens": {"token_ids": description["tokens"]["token_ids"][index],
                                  "mask": description["tokens"]["mask"][index],
                                  "type_ids": description["tokens"]["type_ids"][index]}}
                self._cwe_tree[meta["cwe_id"]] = {"description": des, "children": meta["children"]}
        else:
            # for testing
            # we directly store the embeddings
            desciprtion_embed = self._text_field_embedder(description)
            desciprtion_embed = self._pooler(desciprtion_embed)
            for index, meta in enumerate(metadata):
                self._cwe_tree[meta["cwe_id"]] = {"description_embed": desciprtion_embed[index].unsqueeze(0), "children": meta["children"]}
    
    def beam_predict(self,
                diff_embedding: torch.Tensor,
                diff_mask: torch.BoolTensor,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        # perform beam predict for a test sample
        if len(metadata) != 1:
            raise ValueError(f"beam_predict can only be performed with batch_size=1, got batch_size={len(metadata)}")

        max_depth = min(self._max_depth, len(metadata[0]["instance"]["path"]))  # always self._max_depth, because we already filter samples whose CWE are at depth-1/2

        shape = diff_embedding.shape
        diff_hunk_mask = diff_mask.float().sum(dim=-1) > 0

        diff_embedding_pooled_list = [None]*max_depth
        for depth in range(max_depth):
            if depth != 0:
                # task-specific bi-lstm encoders
                encoder = getattr(self, f"_{depth}_encoder")
                diff_embedding = encoder(inputs=diff_embedding.view(-1, shape[-2], shape[-1]), mask=diff_mask.view(-1, shape[-2]))
                diff_embedding = diff_embedding.view(shape[0], shape[1], shape[2], -1)
            
            diff_embedding_pooled = pooling_with_mask(input=diff_embedding.view(-1, shape[-2], shape[-1]), mask=diff_mask.view(-1, shape[-2]))
            diff_embedding_pooled = pooling_with_mask(input=diff_embedding_pooled.view(shape[0], shape[1], -1), mask=diff_hunk_mask)
            
            diff_embedding_pooled_list[depth] = diff_embedding_pooled
            
        # the first level has no constraints
        logits = self._0_projector(diff_embedding_pooled_list[0])
        log_probs = nn.functional.log_softmax(logits, dim=-1)[0]  # use log, which is more stable
        k_ = min(self._beam_size, log_probs.shape[-1])
        log_probs, indexes = torch.topk(log_probs, k=k_, largest=True, sorted=True)

        queue = list()
        for lprob, idx in zip(log_probs.tolist(), indexes.tolist()):
            queue.append({"prob": lprob, "path": [idx]})

        best_node = [None]*self._top_k  # only the (max_depth-1)
        flag = False
        while len(queue) != 0:
            node = queue.pop(0)

            current_level = len(node["path"])
            
            if current_level == max_depth:
                if best_node[-1] is None:
                    for k in range(self._top_k):
                        if best_node[k] is None:
                            best_node[k] = deepcopy(node)
                            break
                
                if best_node[-1] is not None:
                    flag = True
                    break

                continue

            pre_cwe = self._labels[current_level-1][node["path"][-1]]
            
            projector = getattr(self, f"_{current_level}_projector")

            # chain: use gated sum
            merger = getattr(self, f"_{current_level}_merger")
            merged_embedding = merger(input_a=diff_embedding_pooled_list[current_level], input_b=self._cwe_tree[pre_cwe]["description_embed"])  # be careful with the order
            logits = projector(merged_embedding)

            log_probs = nn.functional.log_softmax(logits, dim=-1)[0]

            valid_children = list()
            for child in self._cwe_tree[pre_cwe]["children"]:
                if child in self._labels[current_level]:
                    # always true
                    valid_children.append(self._labels[current_level].index(child))
            if len(valid_children) == 0:
                # incomplete path
                continue
            
            log_probs = log_probs.tolist()
            lprob_idx = [(log_probs[idx], idx) for idx in valid_children]
            lprob_idx.sort(key=lambda x:x[0], reverse=True)

            for lprob, idx in lprob_idx[:self._beam_size]:
                new_node = deepcopy(node)
                new_node["prob"] += lprob
                new_node["path"].append(idx)
                queue.append(new_node)

            queue.sort(key=lambda x:x["prob"], reverse=True)  # a PriorityQueue (pathes with higher probabilities are first considered)
        
        if not flag:
            raise ValueError(f"can't find any complete path, try larger beam size. [{metadata[0]['instance']['commit_id']}]")

        return best_node


    def forward(self,
                diff: TextFieldTensors,
                metadata: List[Dict[str, Any]] = None,
                **labels) -> Dict[str, Any]:

        output_dict = dict()
        if metadata:
            output_dict["meta"] = metadata
        
        diff_mask = get_text_field_mask(diff, num_wrapping_dims=1, padding_id=1)
        diff_hunk_mask = diff_mask.float().sum(dim=-1) > 0
    
        # sequence field
        diff_embedding = self._text_field_embedder(diff, num_wrapping_dims=1)
        shape = diff_embedding.shape
        
        if metadata and metadata[0]["type"] in ["test", "validation"]:
            result = self.beam_predict(diff_embedding=diff_embedding, diff_mask=diff_mask, metadata=metadata)

            depth = (self._max_depth - 1)
            self._custom_metrics(predictions=[self._labels[depth][result[0]["path"][-1]]], metadata=metadata)
            
            output_dict["meta"][0]["instance"][f"predict_{depth}"] = [self._labels[depth][result[j]["path"][-1]] for j in range(len(result))]
            output_dict["meta"][0]["instance"][f"prob_{depth}"] = [result[j]["prob"] for j in range(len(result))]

            return output_dict

        loss = list()
        for depth in range(self._max_depth):

            if depth != 0:
                # task-specific bi-lstm encoders
                encoder = getattr(self, f"_{depth}_encoder")
                diff_embedding = encoder(inputs=diff_embedding.view(-1, shape[-2], shape[-1]), mask=diff_mask.view(-1, shape[-2]))
                diff_embedding = diff_embedding.view(shape[0], shape[1], shape[2], -1)
            
            diff_embedding_pooled = pooling_with_mask(input=diff_embedding.view(-1, shape[-2], shape[-1]), mask=diff_mask.view(-1, shape[-2]))
            diff_embedding_pooled = pooling_with_mask(input=diff_embedding_pooled.view(shape[0], shape[1], -1), mask=diff_hunk_mask)

            # generate label embedding for chain model
            if depth != 0:
                # schedule sampling
                if random.choices([True, False], weights=[self._teacher_forcing_ratio, 1 - self._teacher_forcing_ratio], k=1)[0]:
                    # use teacher forcing
                    pre_cwe = labels[f"label_{depth-1}"]

                pre_cwe = [self._labels[depth-1][idx] for idx in pre_cwe.tolist()]
                
                cwe_token = {"tokens": {"token_ids": torch.stack([self._cwe_tree[cwe]["description"]["tokens"]["token_ids"] for cwe in pre_cwe]),
                                        "mask": torch.stack([self._cwe_tree[cwe]["description"]["tokens"]["mask"] for cwe in pre_cwe]),
                                        "type_ids": torch.stack([self._cwe_tree[cwe]["description"]["tokens"]["type_ids"] for cwe in pre_cwe])}}
                cwe_embedding = self._text_field_embedder(cwe_token)
                cwe_embedding = self._pooler(cwe_embedding)

                # gated sum
                merger = getattr(self, f"_{depth}_merger")
                diff_embedding_pooled = merger(input_a=diff_embedding_pooled, input_b=cwe_embedding)  # be careful with the order
            
            projector = getattr(self, f"_{depth}_projector")
            logits = projector(diff_embedding_pooled)
            probs = nn.functional.softmax(logits, dim=-1)
            output_dict[f"probs_{depth}"] = probs

            pre_cwe = torch.argmax(probs, -1)  # schedule sampling

            loss.append(self._loss(logits, labels[f"label_{depth}"]))
        
        loss = torch.stack(loss)
        loss = loss * self._depth_weights
        loss = torch.sum(loss)

        output_dict['loss'] = loss

        # just to show the optimization trend of training
        self._metrics[f"depth-0_accuracy"](predictions=output_dict[f"probs_0"], gold_labels=labels[f"label_0"])
        self._metrics[f"depth-0_fscore_weighted"](predictions=output_dict[f"probs_0"], gold_labels=labels[f"label_0"])
        
        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        out2file = list()
        if output_dict["meta"][0]["type"] in ["test", "validation"]:
            out2file.append(output_dict["meta"][0]["instance"])
            return out2file
        
        return out2file

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        
        try:
            metrics[f"depth-0_accuracy"] = self._metrics[f"depth-0_accuracy"].get_metric(reset)
            precision, recall, fscore = self._metrics[f"depth-0_fscore_weighted"].get_metric(reset).values()
            metrics[f'depth-0_weighted_precision'] = precision
            metrics[f'depth-0_weighted_recall'] = recall
            metrics[f'depth-0_weighted_fscore'] = fscore
        except Exception:
            pass
        
        result = self._custom_metrics.get_metric(reset)
        for sub_metric_name, v in result.items():
            metrics[sub_metric_name] = v
        
        return metrics