'''
    Get the PatchCPG dataset from public dataset or from local dataset.
'''

import os
import numpy as np
import shutil
import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip

class PairData(Data):
    def __init__(self, edge_index_s, edge_attr_s, x_s, edge_index_t, edge_attr_t, x_t, y):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

class PatchCPGDataset(Dataset):
    '''
    Reserved for building public dataset.
    Need to modify url, name, raw_file_names, processed_file_names in the future.
    '''

    # download link of the raw numpy dataset.  ## need to modify.
    url = 'https://github.com/shuwang127/shuwang127.github.io/raw/master/'

    def __init__(self, root='./tmp/', transform=None, pre_transform=None):
        self.name = 'PatchCPG'      # downloaded file name.  ## need to modify.
        super(PatchCPGDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # return the file list of self.raw_dir.  ## need to modify.
        return ['data_{}.npz'.format(i) for i in range(8)]

    @property
    def processed_file_names(self):
        # return the file list of self.processed_dir.  ## need to modify.
        return ['data_{}.pt'.format(i) for i in range(8)]

    def download(self):
        # Download to self.raw_dir.
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(os.path.join(self.root, self.name), self.raw_dir)

        return True

    def process(self):
        # process data in self.raw_dir and save to self.processed_dir.
        i = 0
        for raw_path in self.raw_paths:
            graph = np.load(raw_path)
            edgeIndex0 = torch.tensor(graph['edgeIndex0'], dtype=torch.long)
            edgeIndex1 = torch.tensor(graph['edgeIndex1'], dtype=torch.long)
            edgeAttr0 = torch.tensor(graph['edgeAttr0'], dtype=torch.float)
            edgeAttr1 = torch.tensor(graph['edgeAttr1'], dtype=torch.float)
            nodeAttr0 = torch.tensor(graph['nodeAttr0'], dtype=torch.float)
            nodeAttr1 = torch.tensor(graph['nodeAttr1'], dtype=torch.float)
            label = torch.tensor(graph['label'], dtype=torch.long)
            data = PairData(edge_index_s=edgeIndex0, x_s=nodeAttr0, edge_attr_s=edgeAttr0,
                            edge_index_t=edgeIndex1, x_t=nodeAttr1, edge_attr_t=edgeAttr1, y=label)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        # reture the total number of processed samples.
        return len(self.processed_file_names)

    def get(self, idx):
        # get the idx-th sample from self.processed_dir.
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

def GetDataset(path=None):
    '''
    Get the dataset from numpy data files.
    :param path: the path used to store numpy dataset.
    :return: dataset - list of torch_geometric.data.Data
    '''

    # check.
    if None == path:
        print('[ERROR] <GetDataset> The method is missing an argument \'path\'!')
        return []

    # contruct the dataset.
    dataset = []
    files = []
    for root, _, filelist in os.walk(path):
        for file in filelist:
            if not file.endswith('.npz'): continue
            # read a numpy graph file.
            graph = np.load(os.path.join(root, file), allow_pickle=True)
            files.append(file)
            # sparse each element.
            edgeIndex0 = torch.tensor(graph['edgeIndex0'], dtype=torch.long)
            edgeIndex1 = torch.tensor(graph['edgeIndex1'], dtype=torch.long)
            edgeAttr0 = torch.tensor(graph['edgeAttr0'], dtype=torch.float)
            edgeAttr1 = torch.tensor(graph['edgeAttr1'], dtype=torch.float)
            nodeAttr0 = torch.tensor(graph['nodeAttr0'], dtype=torch.float)
            nodeAttr1 = torch.tensor(graph['nodeAttr1'], dtype=torch.float)
            label = torch.tensor(graph['label'], dtype=torch.long)
            # construct an instance of torch_geometric.data.Data.
            data = PairData(edge_index_s=edgeIndex0, edge_attr_s=edgeAttr0, x_s=nodeAttr0,
                            edge_index_t=edgeIndex1, edge_attr_t=edgeAttr1, x_t=nodeAttr1, y=label)
            # append the Data instance to dataset.
            dataset.append(data)

    if (0 == len(dataset)):
        print(f'[ERROR] Fail to load data from {path}')

    return dataset, files