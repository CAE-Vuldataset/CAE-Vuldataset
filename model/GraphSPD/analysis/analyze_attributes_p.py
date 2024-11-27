import os
import numpy as np
import torch

fpath = './data_np/positives/'
filename = './logs/positives.csv'

fp = open(filename, 'w')
for root, ds, fs in os.walk(fpath):
    for file in fs:
        graph = np.load(os.path.join(fpath, file), allow_pickle=True)
        nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
        edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)

        cntA_node = len([attr for attr in nodeAttr if (attr[0] == 1)])
        cntC_node = len([attr for attr in nodeAttr if (attr[0] == 0)])
        cntB_node = len([attr for attr in nodeAttr if (attr[0] == -1)])
        cntB_edge = len([attr for attr in edgeAttr if (attr[0] == 1)])
        cntA_edge = len([attr for attr in edgeAttr if (attr[1] == 1)])
        cntC_edge = len([attr for attr in edgeAttr if (attr[2] == 1)])
        cntD_edge = len([attr for attr in edgeAttr if (attr[3] == 1)])

        print(cntA_node, cntC_node, cntB_node, cntB_edge, cntA_edge, cntC_edge, cntD_edge)
        fp.write(str(cntA_node) + ',' + str(cntC_node) + ',' + str(cntB_node) + ',')
        fp.write(str(cntB_edge) + ',' + str(cntA_edge) + ',' + str(cntC_edge) + ',' + str(cntD_edge))
        fp.write('\n')
fp.close()
