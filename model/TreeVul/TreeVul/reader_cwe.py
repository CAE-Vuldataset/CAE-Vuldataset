import json
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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("reader_cwe")
class ReaderCWE(DatasetReader):
    '''
    support label embedding
    '''
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()

        self._token_indexers = token_indexers  # token indexers for text
        self._tokenizer = tokenizer
        self._tokenizer.sequence_pair_mid_tokens = copy.deepcopy(self._tokenizer.sequence_pair_end_tokens)  # the original one is wrong

    @overrides
    def _read(self, file_path):
        cwe_tree = json.load(open(file_path, 'r'))
        node_count = 0
        for k, v in cwe_tree.items():
            node_count += 1
            yield self.text_to_instance(description = self._tokenizer.tokenize(v["description"]), 
                                        cwe_id = k,
                                        children = v["children"])

        logger.info(f"Num of CWE Nodes is {node_count}")

    @overrides
    def text_to_instance(self, description, cwe_id, children) -> Instance:
        fields: Dict[str, Field] = {}

        fields["description"] = TextField(description, self._token_indexers)
        fields['metadata'] = MetadataField({"cwe_id": cwe_id, "children": children})

        return Instance(fields)