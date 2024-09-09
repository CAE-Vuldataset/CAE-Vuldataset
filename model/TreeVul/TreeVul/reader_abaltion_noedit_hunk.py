import json
import random
from typing import Dict, List, Optional
import logging

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from numpy.lib.function_base import append

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
import pandas as pd
import copy

logger = logging.getLogger(__name__)

@DatasetReader.register("reader_ablation_noedit_hunk")
class ReaderAblationNoEdit(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_file: str = 'valid_cwes.json',
                 invalid_label_index: int = -1,
                 max_depth: int = 3,
                 max_hunk_num: int = 8) -> None:
        super().__init__()

        self._token_indexers = token_indexers  # token indexers for text
        self._tokenizer = tokenizer
        self._tokenizer.sequence_pair_mid_tokens = copy.deepcopy(self._tokenizer.sequence_pair_end_tokens)  # the default SEP is wrong
        
        self._max_depth = max_depth
        self._max_hunk_num = max_hunk_num
        
        self._labels = json.load(open(label_file, 'r'))
        self._invalid_label_index = invalid_label_index

    def read_dataset(self, file_path):
        samples = json.load(open(file_path, 'r', encoding='utf-8'))

        logger.info(f"{file_path} sample num (file-level): {len(samples)}")
        samples_commit = dict()  # commit-level sample
        for s in samples:
            commit_id = s["commit_id"]
            if commit_id not in samples_commit:
                samples_commit[commit_id] = {"commit_id": commit_id, "path": s["path_list"][0], "model_input_pair": []}
            
            # hunk-level input
            for rem_diff, add_diff in zip(s["REM_DIFF"], s["ADD_DIFF"]):

                if len(samples_commit[commit_id]["model_input_pair"]) >= self._max_hunk_num:
                    break

                # preprocess code exactly the same as the CodeBERT
                for s in ['\r\n', '\r', '\n']:
                    rem_diff = rem_diff.replace(s, ' ')
                    add_diff = add_diff.replace(s, ' ')
                rem_diff = ' '.join(rem_diff.strip().split())
                add_diff = ' '.join(add_diff.strip().split())
            
                rem_diff = self._tokenizer.tokenize(rem_diff)
                add_diff = self._tokenizer.tokenize(add_diff)

                diff_pair = (
                    self._tokenizer.sequence_pair_start_tokens +
                    rem_diff +
                    self._tokenizer.sequence_pair_mid_tokens +
                    add_diff +
                    self._tokenizer.sequence_pair_end_tokens
                    )

                samples_commit[commit_id]["model_input_pair"].append(diff_pair)
        
        samples_commit = list(samples_commit.values())

        logger.info(f"[{file_path}] sample num (commit-level): {len(samples_commit)}")

        label_distribution = [dict() for i in range(self._max_depth)]
        for s in samples_commit:
            for depth, cwe_id in enumerate(s["path"]):
                if depth < self._max_depth:
                    if cwe_id not in label_distribution[depth]:
                        label_distribution[depth][cwe_id] = 0
                    label_distribution[depth][cwe_id] += 1
        
        for depth, distribution in enumerate(label_distribution):
            logger.info(f"label distribution [depth-{depth}]: {distribution}")

        return samples_commit

    @overrides
    def _read(self, file_path):
        dataset = self.read_dataset(file_path)

        if "test" in file_path:
            logger.info("loading testing samples ...")
            num_sample = 0
            for sample in dataset:
                yield self.text_to_instance(ins=sample, type_="test")
                num_sample += 1

            logger.info(f"Num of testing instances is [{num_sample}]")

        elif "validation" in file_path:
            logger.info("loading validation examples ...")
            num_sample = 0
            for sample in dataset:
                yield self.text_to_instance(ins=sample, type_="validation")
                num_sample += 1

            logger.info(f"Num of validation instances is [{num_sample}]")
            
        else:
            # training
            logger.info("loading training examples ...")
            num_sample = 0
            for sample in dataset:
                yield self.text_to_instance(ins=sample, type_="train")
                num_sample += 1

            logger.info(f"Num of training instances is [{num_sample}]")

    @overrides
    def text_to_instance(self, ins, type_="train") -> Instance:
        fields: Dict[str, Field] = {}
        
        fields["diff"] = ListField([TextField(pair, self._token_indexers) for pair in ins["model_input_pair"]])  # hunk-level dataset

        for depth in range(self._max_depth):
            if type_ == "test" and ins["path"][depth] not in self._labels[depth]:
                # in test set, it is possible to have new cwes
                label_index = self._invalid_label_index
            else:
                label_index = self._labels[depth].index(ins["path"][depth])

            fields[f"label_{depth}"] = LabelField(label=label_index, label_namespace=f"depth_{depth}_labels", skip_indexing=True)

        meta_ins = {"commit_id": ins["commit_id"], "path": ins["path"]}
        fields['metadata'] = MetadataField({"type": type_, "instance": meta_ins})

        return Instance(fields)