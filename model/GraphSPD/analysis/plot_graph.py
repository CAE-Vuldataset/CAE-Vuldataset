'''
    Visualize the Patch-CPG.
'''

import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
from libs.graphvis import VisualGraph
import argparse

np.set_printoptions(threshold=np.inf)

rootPath = './'
tempPath = './temp/'
rdatPath = rootPath + '/data_raw/'
ndatPath = rootPath + '/data_np/'
ndatFile = ''
saveFile = ''

_SHOWMAP = True
_SHOWFIG = False
_OPTION = 0
_STATE = 0

def main():
    if os.path.exists(ndatFile):
        ndatFilePath = ndatFile
    elif ndatFile in os.listdir(os.path.join(ndatPath, 'positives')):
        ndatFilePath = os.path.join(ndatPath + 'positives', ndatFile).replace('\\', '/')
    elif ndatFile in os.listdir(os.path.join(ndatPath, 'negatives')):
        ndatFilePath = os.path.join(ndatPath + 'negatives', ndatFile).replace('\\', '/')
    else:
        print('[ERROR] Cannot find the file ', ndatFile)
        return

    if _SHOWMAP:
        FindCodeMapping(ndatFilePath)

    print('[INFO] Read graph file from ', ndatFilePath)
    graph = ReadGraph(ndatFilePath)
    graphfig = VisualGraph(graph, state=_STATE, options=_OPTION, show_label=_SHOWFIG)

    # save figure.
    if (0 == len(saveFile)):
        pathseg = ndatFilePath.split(sep='/')
        filename = pathseg[-1][:-4] + '_op' + str(_OPTION) + '_st' + str(_STATE) + '.png'
        filename = os.path.join(tempPath, filename).replace('\\', '/')
        if not os.path.exists(tempPath):
            os.mkdir(tempPath)
    else:
        filename = saveFile

    print('[INFO] Save the figure in ', filename)
    graphfig.savefig(filename, dpi=graphfig.dpi)

    return

def FindCodeMapping(fp):

    fp_split = fp.split('/')
    filename = fp_split[-1][:-4] + '.log'
    if filename in os.listdir(os.path.join(rdatPath, 'positives')):
        filepath = os.path.join(rdatPath + 'positives', filename).replace('\\', '/')
    elif filename in os.listdir(os.path.join(rdatPath, 'negatives')):
        filepath = os.path.join(rdatPath + 'negatives', filename).replace('\\', '/')
    else:
        print('[WARNING] Cannot find the file ', filename)
        return

    g = np.load(os.path.join(fp), allow_pickle=True)
    nodeDict = g['nodeDict'].item()
    mapDict = {node: id for node, id in enumerate(nodeDict)}

    f = open(filepath, encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()

    codeDict = dict()
    signEdge = 1
    for line in lines:
        if line.startswith('==='):
            signEdge = 0
        elif (0 == signEdge):
            # Node:
            contents = re.findall(r'\(-\d+, [\'\"]\(.*\)[\'\"], -?\d\)', line)
            if 0 == len(contents):
                continue
            content = contents[0]  # get the first match.
            content = content[1:-1]  # remove ().
            segs = content.split(',')  # split with comma.
            attr = ','.join(segs[1:-1])  # list attribute.
            attrSegs = attr[3:-2].split('),(')
            attrList = [seg.split(',', maxsplit=1) for seg in attrSegs]
            attrCode = ''
            for attr in attrList:
                if attr[0] == 'CODE':
                    attrCode = attr[1]
            codeDict[segs[0]] = attrCode

    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    pSave = os.path.join(tempPath, fp_split[-1][:-4] + '.txt')
    fSave = open(pSave, 'w')

    retDict = dict()
    for i in range(len(mapDict)):
        retDict[i] = codeDict[mapDict[i]]
        print(i, ' ', codeDict[mapDict[i]])
        fSave.write(str(i) + ',' + codeDict[mapDict[i]] + '\n')
    fSave.close()
    print('[INFO] Save the mapping dictionary to ', pSave)

    return retDict

def ReadGraph(filename):

    graph = np.load(os.path.join(filename), allow_pickle=True)
    # sparse each element.
    edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
    nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
    edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)
    label = torch.tensor(graph['label'], dtype=torch.long)
    # construct an instance of torch_geometric.data.Data.
    data = Data(edge_index=edgeIndex, x=nodeAttr, edge_attr=edgeAttr, y=label)

    return data

def ArgsParser():
    # define argument parser.
    parser = argparse.ArgumentParser()
    # add arguments.
    parser.add_argument('-npz', help='the graph npz file name.', required=True)
    parser.add_argument('-save', help='the graph save file path.')
    parser.add_argument('-option', help='the edge option of graph. (0:version; 1:edgetype)', type=int)
    parser.add_argument('-state', help='the init state of graph.', type=int)
    parser.add_argument('-show', help='show the figure', action='store_true')
    # parse the arguments.
    args = parser.parse_args()
    # global variables.
    global ndatFile
    global _SHOWFIG
    global saveFile
    global _STATE
    global _OPTION
    # perform actions.
    if (args.npz): ndatFile = args.npz
    if (args.show): _SHOWFIG = True
    if (args.save): saveFile = args.save
    if (args.state): _STATE = args.state
    if (args.option): _OPTION = args.option

    return parser

if __name__ == '__main__':
    # sparse the arguments.
    ArgsParser()
    # main
    main()