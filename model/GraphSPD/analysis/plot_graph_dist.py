import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rootPath = './'
tempPath = './temp/'
ndatPath = rootPath + '/data_np/'

def main():
    records = GetRecords()

    fig = plt.figure(1)

    NODEMAX = 1000
    NODESTP = 25
    EDGEMAX = 1000
    EDGESTP = 25

    plt.subplot(3, 2, 1)
    nRecList = records.loc[records['label'] <= 0]
    nRecList = nRecList['nodeNum'].tolist()
    nRecList = [min(nRec, NODEMAX) for nRec in nRecList]
    binList = [i for i in range(0, NODEMAX, NODESTP)]
    plt.hist(nRecList, bins=binList, facecolor="blue", edgecolor="black", alpha=0.7)

    plt.subplot(3, 2, 3)
    nRecList = records.loc[records['label'] >= 1]
    nRecList = nRecList['nodeNum'].tolist()
    nRecList = [min(nRec, NODEMAX) for nRec in nRecList]
    binList = [i for i in range(0, NODEMAX, NODESTP)]
    plt.hist(nRecList, bins=binList, facecolor="blue", edgecolor="black", alpha=0.7)

    plt.subplot(3, 2, 5)
    nRecList = records.loc[:, ['nodeNum']]
    nRecList = nRecList['nodeNum'].tolist()
    nRecList = [min(nRec, NODEMAX) for nRec in nRecList]
    binList = [i for i in range(0, NODEMAX, NODESTP)]
    plt.hist(nRecList, bins=binList, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel('#nodes')

    plt.subplot(3, 2, 2)
    nRecList = records.loc[records['label'] <= 0]
    nRecList = nRecList['edgeNum'].tolist()
    nRecList = [min(nRec, EDGEMAX) for nRec in nRecList]
    binList = [i for i in range(0, EDGEMAX, EDGESTP)]
    plt.hist(nRecList, bins=binList, facecolor="blue", edgecolor="black", alpha=0.7)

    plt.subplot(3, 2, 4)
    nRecList = records.loc[records['label'] >= 1]
    nRecList = nRecList['edgeNum'].tolist()
    nRecList = [min(nRec, EDGEMAX) for nRec in nRecList]
    binList = [i for i in range(0, EDGEMAX, EDGESTP)]
    plt.hist(nRecList, bins=binList, facecolor="blue", edgecolor="black", alpha=0.7)

    plt.subplot(3, 2, 6)
    nRecList = records.loc[:, ['edgeNum']]
    nRecList = nRecList['edgeNum'].tolist()
    nRecList = [min(nRec, EDGEMAX) for nRec in nRecList]
    binList = [i for i in range(0, EDGEMAX, EDGESTP)]
    plt.hist(nRecList, bins=binList, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel('#edges')

    plt.show()

    fig.savefig(tempPath + 'records.png', dpi=fig.dpi)

    return

def GetRecords():

    if os.path.exists(tempPath + 'records.csv'):
        records = pd.read_csv(tempPath + 'records.csv')
        return records

    records = pd.DataFrame(columns=['filename', 'nodeNum', 'edgeNum', 'label'])
    for root, ds, fs in os.walk(ndatPath):
        for f in fs:
            npzname = os.path.join(root, f).replace('\\', '/')
            nNode, nEdge, label = GraphStats(npzname)
            records = records.append(pd.DataFrame({'filename': [f], 'nodeNum': [nNode],
                                                   'edgeNum': [nEdge], 'label': [label]}), ignore_index=True)

    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    records.to_csv(tempPath + 'records.csv', index=0)

    return records

def GraphStats(filename):
    graph = np.load(os.path.join(filename), allow_pickle=True)
    # sparse each element.
    nodeAttr = graph['nodeAttr']
    edgeAttr = graph['edgeAttr']
    label = graph['label']

    return  len(nodeAttr), len(edgeAttr), label[0]

if __name__ == '__main__':
    main()