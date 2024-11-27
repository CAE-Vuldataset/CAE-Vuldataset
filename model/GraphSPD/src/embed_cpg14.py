import os
import urllib
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel

def SetupConfigs(opt):
    # set configuration.
    if 0 == opt.embed_config: 
        opt.conf_path = os.path.join(opt.root, 'configs/codebert-base/')
        url = 'https://huggingface.co/microsoft/codebert-base/resolve/main/pytorch_model.bin'
    elif 1 == opt.embed_config:
        opt.conf_path = os.path.join(opt.root, 'configs/codebert-cpp/')
        url = 'https://huggingface.co/neulab/codebert-cpp/resolve/main/pytorch_model.bin'
    # download the large model.
    model_file = os.path.join(opt.conf_path, 'pytorch_model.bin')
    if not os.path.exists(model_file):
        urllib.request.urlretrieve(url, model_file)
    # load tokenizer and model.
    opt.tokenizer = RobertaTokenizer.from_pretrained(opt.conf_path)
    opt.embed_model = RobertaModel.from_pretrained(opt.conf_path)
    opt.embed_model.to(torch.device("cpu"))
    return 0

def ValEmbedCPG14(opt, commitID):
    # get the src and dst file path.
    filename = os.path.join(opt.vmid_path, commitID+'.npz') 
    savename = os.path.join(opt.vnp_path, commitID+'.npz')
    os.makedirs(os.path.join(opt.vnp_path), exist_ok=True)
    # read the edges and nodes.
    nodes, edges, label = ReadFile(filename)

    ## process the patch graph.
    nodeDict, edgeIndex, edgeAttr = ProcEdges(edges[0])
    nodeAttr, err = ProcNodes(opt, nodes[0], nodeDict)
    if err: # err.
        return 1
    else: # ok.
        np.savez(savename, edgeIndex=edgeIndex, edgeAttr=edgeAttr, 
                 nodeAttr=nodeAttr, label=label, nodeDict=nodeDict) # no need to save dict.

    return 0

def ValEmbedCPG14Twin(opt, commitID):
    # get the src and dst file path.
    filename = os.path.join(opt.vmid_path, commitID+'.npz')
    savename2 = os.path.join(opt.vnp2_path, commitID+'.npz')
    os.makedirs(os.path.join(opt.vnp2_path), exist_ok=True)
    # read the edges and nodes.
    nodes, edges, label = ReadFile(filename)

    ## process the twin graph.
    nodeDictA, edgeIndexA, edgeAttrA = ProcEdges(edges[1])
    nodeAttrA, errA = ProcNodes(opt, nodes[1], nodeDictA)
    nodeDictB, edgeIndexB, edgeAttrB = ProcEdges(edges[2])
    nodeAttrB, errB = ProcNodes(opt, nodes[2], nodeDictB)
    if errA or errB: # err.
        return 1
    else: # ok.
        np.savez(savename2, edgeIndex0=edgeIndexA, edgeAttr0=edgeAttrA, nodeAttr0=nodeAttrA, 
                 edgeIndex1=edgeIndexB, edgeAttr1=edgeAttrB, nodeAttr1=nodeAttrB, label=label)
        
    return 0

def EmbedCPG14(opt, commitID):
    # get the src and dst file path.
    filename = os.path.join(opt.mid_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tmid_path, commitID+'.npz')
    savename = os.path.join(opt.np_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tnp_path, commitID+'.npz')
    # read the edges and nodes.
    nodes, edges, label = ReadFile(filename)

    ## process the patch graph.
    nodeDict, edgeIndex, edgeAttr = ProcEdges(edges[0])
    nodeAttr, err = ProcNodes(opt, nodes[0], nodeDict)
    if err: # err.
        return 1
    else: # ok.
        np.savez(savename, edgeIndex=edgeIndex, edgeAttr=edgeAttr, 
                 nodeAttr=nodeAttr, label=label, nodeDict=nodeDict) # no need to save dict.

    return 0

def EmbedCPG14Twin(opt, commitID):
    # get the src and dst file path.
    filename = os.path.join(opt.mid_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tmid_path, commitID+'.npz')
    savename2 = os.path.join(opt.np2_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tnp2_path, commitID+'.npz')
    # read the edges and nodes.
    nodes, edges, label = ReadFile(filename)

    ## process the twin graph.
    nodeDictA, edgeIndexA, edgeAttrA = ProcEdges(edges[1])
    nodeAttrA, errA = ProcNodes(opt, nodes[1], nodeDictA)
    nodeDictB, edgeIndexB, edgeAttrB = ProcEdges(edges[2])
    nodeAttrB, errB = ProcNodes(opt, nodes[2], nodeDictB)
    if errA or errB: # err.
        return 1
    else: # ok.
        np.savez(savename2, edgeIndex0=edgeIndexA, edgeAttr0=edgeAttrA, nodeAttr0=nodeAttrA, 
                 edgeIndex1=edgeIndexB, edgeAttr1=edgeAttrB, nodeAttr1=nodeAttrB, label=label)
        
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
    print(f'[INFO] <ProcEdges> Find {len(nodeDict)} nodes connected with {len(edgesData)} edges.')

    # get the edge index. [2 * edge_num]
    nodesoutIndex = [nodeDict[node] for node in nodesout]
    nodesinIndex = [nodeDict[node] for node in nodesin]
    edgeIndex = np.array([nodesoutIndex, nodesinIndex])
    print(f'[INFO] <ProcEdges> Get {len(edgeIndex)} * {len(edgeIndex[0])} edge index array.')

    ## EDGE EMBEDDING.
    # get the dictionary of version and type.
    verDict = {'-1': [1, 0], '0': [1, 1], '1': [0, 1], 
               -1: [1, 0], 0: [1, 1], 1: [0, 1]}
    typeDict = {'CFG': [1, 0, 0], 'CDG': [1, 0, 0], 'DDG': [0, 1, 0], 'AST': [0, 0, 1]}
    # get the edge attributes. [edge_num, num_edge_features]
    typeAttr = np.array([typeDict[edge[2][:3]] for edge in edgesData])
    verAttr = np.array([verDict[edge[3]] for edge in edgesData])
    edgeAttr = np.c_[verAttr, typeAttr]
    print(f'[INFO] <ProcEdges> Get {len(edgeAttr)} * {len(edgeAttr[0])} edge attribute array.')

    return nodeDict, edgeIndex, edgeAttr

def ProcNodes(opt, nodesData, nodeDict):
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
        return np.zeros((2, opt.embed_dim)), 0  # ok. set 2 void nodes.

    # get the list of all nodes.
    nodeList = [nodeData[0] for nodeData in nodesData]
    # check the integrity of the node list. nodeList should cover nodeDict.
    for node in nodeDict:
        if node not in nodeList:
            print(f'[ERROR] <ProcNodes> Node {node} does not in node list.')
            return np.zeros((2, opt.embed_dim)), 1 # err.

    # get the node attributes with the order of node dictionary.
    nodeOrder = [nodeList.index(node) for node in nodeDict]
    nodesDataNew = [nodesData[order] for order in nodeOrder]
    if (0 == len(nodeDict)): # if #edges=0 but #nodes>0, construct 2-node graph with a all-0-edge.
        nodesDataNew = [nodesData[0], nodesData[0] if len(nodesData)==1 else nodesData[1]]

    ## NODE EMBEDDING.
    nodeAttr = []
    for nodeData in nodesDataNew:
        tokens = opt.tokenizer.tokenize(nodeData[-1]) # tokenization.
        if len(tokens):
            tokens_ids = opt.tokenizer.convert_tokens_to_ids(tokens) # to token ids.
            tokens_ids = tokens_ids[:512] if len(tokens_ids) > 512 else tokens_ids # truncate if necessary.
            embeds = opt.embed_model(torch.tensor(tokens_ids)[None,:])[1] # get embeddings
            embeds = torch.mean(embeds, dim=1) if embeds.dim() == 3 else embeds # avg pooling if necessary.
            embeds = torch.flatten(embeds).detach().numpy() # to numpy
        else:
            embeds = np.zeros(opt.embed_dim)
        # append the node embedding.
        nodeAttr.append(embeds)
    nodeAttr = np.array(nodeAttr)

    print(f'[INFO] <ProcNodes> Get {len(nodeAttr)} * {len(nodeAttr[0])} node attribute array.')

    return nodeAttr, 0 # ok.
