import copy
import json
import logging
import sys
import os
os.chdir(sys.path[0])
import torch
from dgl import DGLGraph
from tqdm import tqdm
from multiprocessing import Pool

from data_loader.batch_graph import GGNNBatchGraph
from utils import load_default_identifiers, initialize_batch, debug

##for each function
class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target):
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})   ##
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})

class DataSet:
    def __init__(self, train_src, valid_src, test_src, batch_size, n_ident=None, g_ident=None, l_ident=None, args=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.args = args
        self.read_dataset(train_src, valid_src, test_src)
        self.initialize_dataset()

    def initialize_dataset(self):

        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def _read_dataset(self, item):
        entry = item
        example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident], edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
        return example

    def read_dataset(self, train_src, valid_src, test_src):
        with open(train_src,"r") as fp:
            train_data = []
            train_data = json.load(fp)
            # train_data = train_data[:10]
            # with Pool(50) as pool:
            #     for example in tqdm(pool.imap_unordered(self._read_dataset, train_data), ncols=100, desc='read train_data by poll', total=len(train_data)):
            #         if self.feature_size == 0:
            #             self.feature_size = example.features.size(1)
            #         self.train_examples.append(example)

            # exit(0)

            for entry in tqdm(train_data, ncols=100, desc='read train_data', mininterval=1):
                example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                    edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                                
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                self.train_examples.append(example)

        if valid_src is not None:
            with open(valid_src,"r") as fp:
                valid_data = []
                valid_data = json.load(fp)
                # valid_data = valid_data[:10] 
                for entry in tqdm(valid_data, ncols=100, desc='read valid_data', mininterval=1):
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.valid_examples.append(example)
        
        if test_src is not None:
            with open(test_src) as fp:
                test_data = []
                test_data = json.load(fp)
                # test_data = test_data[:10]
                for entry in tqdm(test_data, ncols=100, desc='read test_data', mininterval=1):
                    # print(f"label key = {self.l_ident}")
                    # print(f"label = {entry[self.l_ident][0][0]}")
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    if self.feature_size == 0:
                        self.feature_size = example.features.size(1)
                    self.test_examples.append(example)
    
    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=False)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)
        
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:

            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):

        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
