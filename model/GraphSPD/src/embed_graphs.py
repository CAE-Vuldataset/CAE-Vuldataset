'''
    embed the nodes and edges of graph data.
'''

import os
import sys
import time
import urllib
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel

# environment settings.
rootPath = './'
# dictPath = rootPath + '/dict/'
mdatPath = rootPath + '/_data/data_mid/' # mid-point graph folder.
cfg0Path = rootPath + '/configs/codebert-base/' # default model folder.
cfg1Path = rootPath + '/configs/codebert-cpp/' # custom model folder.
tempPath = './'
ndatPath = tempPath + '/_data/data_np/'  # patch graph folder.
ndt2Path = tempPath + '/_data/data_np2/' # twin graph folder.
logsPath = tempPath + '/logs/'     # logs folder.

# hyper-parameters.
_EmbedDim_ = 768
# output parameters.
_DEBUG_  = 0
_ERROR_  = 1
_CONFG_  = 0 # 0: base; 1: custom.
_PATCH_  = 1
_TWINS_  = 1
# global variable.
start_time = time.time() #mark start time

# set configuration.
if _CONFG_: # custom
    confPath = cfg1Path
    url = 'https://huggingface.co/neulab/codebert-cpp/resolve/main/pytorch_model.bin'
else:
    confPath = cfg0Path
    url = 'https://huggingface.co/microsoft/codebert-base/resolve/main/pytorch_model.bin'
# download the large model.
model_file = os.path.join(confPath, 'pytorch_model.bin')
if not os.path.exists(model_file):
    urllib.request.urlretrieve(url, model_file)

# load tokenizer and model.
tokenizer = RobertaTokenizer.from_pretrained(confPath)
model = RobertaModel.from_pretrained(confPath)
model.to(torch.device("cpu"))

class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def RunTime():
    pTime = ' [TIME: ' + str(round((time.time() - start_time), 2)) + ' sec]'
    return pTime

def main():
    cnt, cnt_patch, cnt_twins = 0, 0, 0
    for root, _, fs in os.walk(mdatPath):
        for file in fs:
            if '.DS_Store' in file: continue
            # get the src file and dst file path.
            filename = os.path.join(root, file).replace('\\', '/')
            cnt += 1
            # read the edges and nodes.
            nodes, edges, label = ReadFile(filename)
            ## process the patch graph.
            if _PATCH_:
                savename = filename.replace(mdatPath, ndatPath)
                if os.path.exists(savename):
                    cnt_patch += 1
                    print(f'[INFO] <main> Already found the graph (patch) numpy file: [{cnt_patch}|{cnt}] ' + savename + RunTime())
                else:
                    nodeDict, edgeIndex, edgeAttr = ProcEdges(edges[0])
                    nodeAttr, err = ProcNodes(nodes[0], nodeDict)
                    if err: 
                        print(f'[ERROR] <main> There are error(s) in constructing graph: [{cnt_patch}|{cnt}] ' + savename + RunTime())
                    else: # ok.
                        np.savez(savename, edgeIndex=edgeIndex, edgeAttr=edgeAttr, nodeAttr=nodeAttr, label=label, nodeDict=nodeDict) # no need to save dict.
                        cnt_patch += 1
                        print(f'[INFO] <main> Save the graph information (patch) into numpy file: [{cnt_patch}|{cnt}] ' + savename + RunTime())
                print('-----------------------------------------------------')
            ## process the twin graph.
            if _TWINS_:
                savename2 = filename.replace(mdatPath, ndt2Path)
                if os.path.exists(savename2):
                    cnt_twins += 1
                    print(f'[INFO] <main> Already found the graph (twins) numpy file: [{cnt_twins}|{cnt}] ' + savename2 + RunTime())
                else:
                    nodeDictA, edgeIndexA, edgeAttrA = ProcEdges(edges[1])
                    nodeAttrA, errA = ProcNodes(nodes[1], nodeDictA)
                    nodeDictB, edgeIndexB, edgeAttrB = ProcEdges(edges[2])
                    nodeAttrB, errB = ProcNodes(nodes[2], nodeDictB)
                    if errA or errB:
                        print(f'[ERROR] <main> There are error(s) in constructing graph: [{cnt_twins}|{cnt}] ' + savename2 + RunTime())
                    else: # ok.
                        np.savez(savename2, edgeIndex0=edgeIndexA, edgeAttr0=edgeAttrA, nodeAttr0=nodeAttrA,
                                edgeIndex1=edgeIndexB, edgeAttr1=edgeAttrB, nodeAttr1=nodeAttrB, label=label)
                        cnt_twins += 1
                        print(f'[INFO] <main> Save the graph information (twins) into numpy file: [{cnt_twins}|{cnt}] ' + savename2 + RunTime())
                print('=====================================================')
            # =====================================================
    return 0

def ReadFile(filename):

    graph = np.load(filename, allow_pickle=True)
    nodes = {0: graph['nodesP'], 1: graph['nodesA'], 2: graph['nodesB']}
    edges = {0: graph['edgesP'], 1: graph['edgesA'], 2: graph['edgesB']}
    label = graph['label']

    return nodes, edges, label

def ProcEdges(edgesData):
    '''
    Mapping the edges to edge embeddings.
    :param edgesData: [['-32', '-51', 'EDGE_TYPE', '0'], ...]
    :return: nodeDict - {'-32': 0, '-51': 1, ...}
             edgeIndex - [[0, 1, ...], [1, 2, ...]]
             edgeAttr - [[1, 0, 0, 0, 1], ...]
    '''

    if 0 == len(edgesData): # there is no edge in graph. It can be single-node graph, throw a warning.
        print('[WARNING] <ProcEdges> Find a graph without edges.')
        return {}, np.array([[0], [1]]), np.zeros((1, 5)) # one edge, attr: [0 0 0 0 0]

    # get the node set.
    nodesout = [edge[0] for edge in edgesData]
    nodesin = [edge[1] for edge in edgesData]
    nodeset = nodesout + nodesin
    nodeset = list({}.fromkeys(nodeset).keys()) # remove duplicates
    # get the dictionary.
    nodeDict = {node: index for index, node in enumerate(nodeset)}
    print(f'[INFO] <ProcEdges> Find {len(nodeDict)} nodes connected with {len(edgesData)} edges.' + RunTime())
    if _DEBUG_: print(nodeDict)

    # get the edge index. [2 * edge_num]
    nodesoutIndex = [nodeDict[node] for node in nodesout]
    nodesinIndex = [nodeDict[node] for node in nodesin]
    edgeIndex = np.array([nodesoutIndex, nodesinIndex])
    print(f'[INFO] <ProcEdges> Get {len(edgeIndex)} * {len(edgeIndex[0])} edge index array.' + RunTime())
    if _DEBUG_: print(edgeIndex)

    ## EDGE EMBEDDING.
    # get the dictionary of version and type.
    verDict = {'-1': [1, 0], '0': [1, 1], '1': [0, 1], 
               -1: [1, 0], 0: [1, 1], 1: [0, 1]}
    typeDict = {'CFG': [1, 0, 0], 'CDG': [1, 0, 0], 'DDG': [0, 1, 0], 'AST': [0, 0, 1]}
    # get the edge attributes. [edge_num, num_edge_features]
    typeAttr = np.array([typeDict[edge[2][:3]] for edge in edgesData])
    verAttr = np.array([verDict[edge[3]] for edge in edgesData])
    edgeAttr = np.c_[verAttr, typeAttr]
    print(f'[INFO] <ProcEdges> Get {len(edgeAttr)} * {len(edgeAttr[0])} edge attribute array.' + RunTime())
    if _DEBUG_: print(edgeAttr)

    return nodeDict, edgeIndex, edgeAttr

def ProcNodes(nodesData, nodeDict):
    '''
    Mapping the nodes to node embeddings.
    :param nodesData: [['-165', '0', 'C', '2', '11655', 
                        list(['*', 'ptr', '=', '(', 'delta_base', '<<', '4', ')', '|', 'length_base'])], 
                        ...]
    :param nodeDict: {'-32': 0, '-51': 1, ...}
    :return: [[...], [...], ...]
    '''

    if (0 == len(nodesData)): # if #nodes=0 -> #edges=0.
        print('[WARNING] <ProcNodes> Find a graph without nodes.')
        return np.zeros((2, _EmbedDim_)), 0  # ok. set 2 void nodes.

    # get the list of all nodes.
    nodeList = [nodeData[0] for nodeData in nodesData]
    # check the integrity of the node list. nodeList should cover nodeDict.
    for node in nodeDict:
        if node not in nodeList:
            if _ERROR_: print('[ERROR] <ProcNodes> Node', node, 'does not in node list.')
            return np.zeros((2, _EmbedDim_)), 1 # err.

    # get the node attributes with the order of node dictionary.
    nodeOrder = [nodeList.index(node) for node in nodeDict]
    nodesDataNew = [nodesData[order] for order in nodeOrder]
    if (0 == len(nodeDict)): # if #edges=0 but #nodes>0, construct 2-node graph with a all-0-edge.
        nodesDataNew = [nodesData[0], nodesData[0] if len(nodesData)==1 else nodesData[1]]

    ## NODE EMBEDDING.
    nodeAttr = []
    for nodeData in nodesDataNew:
        tokens = tokenizer.tokenize(nodeData[-1]) # tokenization.
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens) # to token ids.
        tokens_ids = tokens_ids[:512] if len(tokens_ids) > 512 else tokens_ids # truncate if necessary.
        embeds = model(torch.tensor(tokens_ids)[None,:])[1] # get embeddings
        embeds = torch.mean(embeds, dim=1) if embeds.dim() == 3 else embeds # avg pooling if necessary.
        embeds = torch.flatten(embeds).detach().numpy() # to numpy
        # append the node embedding.
        nodeAttr.append(embeds)
    nodeAttr = np.array(nodeAttr)

    print('[INFO] <ProcNodes> Get', len(nodeAttr), '*', len(nodeAttr[0]), 'node attribute array.' + RunTime())

    return nodeAttr, 0 # ok.

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'embed_graphs.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # check folders.
    if not os.path.exists(ndatPath + '/negatives/'):
        os.makedirs(ndatPath + '/negatives/')
    if not os.path.exists(ndatPath + '/positives/'):
        os.makedirs(ndatPath + '/positives/')
    if not os.path.exists(ndt2Path + '/negatives/'):
        os.makedirs(ndt2Path + '/negatives/')
    if not os.path.exists(ndt2Path + '/positives/'):
        os.makedirs(ndt2Path + '/positives/')
    # main entrance.
    main()
