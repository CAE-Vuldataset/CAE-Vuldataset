'''
    Get the PatchCPG dataset from public dataset or from local dataset.
'''

import os
import numpy as np
import shutil
import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip

class PatchCPGDataset(Dataset):
    '''
    Reserved for building public dataset.
    Need to modify url, name, raw_file_names, processed_file_names in the future.
    '''

    # download link of the raw numpy dataset.  ## need to modify.
    url = 'https://github.com/shuwang127/Paimon/raw/master/'

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
            edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
            nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
            edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)
            label = torch.tensor(graph['label'], dtype=torch.long)
            data = Data(edge_index=edgeIndex, x=nodeAttr, edge_attr=edgeAttr, y=label)

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
            edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
            nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
            edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)
            label = torch.tensor(graph['label'], dtype=torch.long)
            # construct an instance of torch_geometric.data.Data.
            data = Data(edge_index=edgeIndex, x=nodeAttr, edge_attr=edgeAttr, y=label)
            # append the Data instance to dataset.
            dataset.append(data)

    if (0 == len(dataset)):
        print(f'[ERROR] Fail to load data from {path}')

    return dataset, files

def GetTrainDataset(path=None, txt=None):
    '''
    Get the dataset from numpy data files.
    :param path: the path used to store numpy dataset.
    :param txt_file: path to the txt file containing filenames.
    :return: dataset - list of torch_geometric.data.Data
             files - list of loaded npz files.
    '''

    # check.
    if path is None:
        print('[ERROR] <GetDataset> The method is missing an argument \'path\'!')
        return [], []
    
    if f'data/{txt}.txt' is None:
        print('[ERROR] <GetDataset> The method is missing an argument \'txt_file\'!')
        return [], []

    # Load file names from the txt file
    valid_files = set()
    with open(f'data/{txt}.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:  # Ensure there are at least 3 parts
                valid_files.add(parts[2])  # Add the filename from the third column

    # print(valid_files)
    # construct the dataset.
    dataset = []
    files = []
    for root, _, filelist in os.walk(path):
        for file in filelist:
            # print(os.path.splitext(file)[0] not in valid_files)
            if not file.endswith('.npz'): 
                continue
            if os.path.splitext(file)[0] not in valid_files:  # Check if the file is in the valid_files set
                continue
            
            # read a numpy graph file.
            graph = np.load(os.path.join(root, file), allow_pickle=True)
            files.append(file)
            # sparse each element.
            edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
            nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
            edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)
            label = torch.tensor(graph['label'], dtype=torch.long)
            # construct an instance of torch_geometric.data.Data.
            data = Data(edge_index=edgeIndex, x=nodeAttr, edge_attr=edgeAttr, y=label)
            # append the Data instance to dataset.
            dataset.append(data)

    if len(dataset) == 0:
        print(f'[ERROR] Fail to load data from {path}')

    return dataset, files