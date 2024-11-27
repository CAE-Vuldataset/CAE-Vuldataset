'''
    Parse the graph from text to numpy objects.
'''

import os
import re
import sys
import time
import numpy as np

# environment settings.
rootPath = './'
dataPath = rootPath + '/_data/data_raw/' # data folders.
tempPath = './'
mdatPath = tempPath + '/_data/data_mid/' # folder to store the middle results.
logsPath = tempPath + '/logs/' # logs folder.

# output parameters.
_DEBUG_  = 0
_ERROR_  = 1
# global variable.
start_time = time.time() #mark start time

# Logger: redirect the stream on screen and to file.
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
    cnt, cnt_save = 0, 0
    for root, _, fs in os.walk(dataPath):
        for file in fs:
            if ('.DS_Store' in file): continue
            # get the paths for src file and dst file.
            filename = os.path.join(root, file).replace('\\', '/')
            subfolder = '/positives/' if ('positives' in root) else '/negatives/'
            savename = os.path.join(mdatPath + subfolder, file[:-4] + '.npz')
            cnt += 1
            if os.path.exists(savename):
                cnt_save += 1
                print(f'[INFO] <main> Already found the graph numpy file: [{str(cnt_save)}|{str(cnt)}] ' + savename + RunTime())
                print('=====================================================')
                continue
            # parse the label and nodes/edges.
            label = [1] if ('positives' in root) else [0]
            nodes, edges, err = ReadFile(filename)
            if err: 
                print(f'[ERROR] <main> There are error(s) in parsing graphs: [{str(cnt_save)}|{str(cnt)}] ' + savename + RunTime())
                print('=====================================================')
                continue
            # save the graph info into the npz file.
            np.savez(savename, nodesP=nodes[0], edgesP=edges[0], nodesA=nodes[1], edgesA=edges[1], nodesB=nodes[2], edgesB=edges[2], label=label, dtype=object)
            cnt_save += 1
            print(f'[INFO] <main> Save the graph information into numpy file: [{str(cnt_save)}|{str(cnt)}] ' + savename + RunTime())
            print('=====================================================')
    return 0

def ParseEdge(filename, line):
    if _DEBUG_: print(line, end='')

    # the graph has no edge.
    if '\n' == line:
        return [], 0 # ok.

    # find the pattern-matched content.
    pattern = r'\(-\d+, -\d+, [\'\"].*[\'\"], -?\d\)'
    contents = re.findall(pattern, line)
    if 0 == len(contents):
        if _ERROR_: print('[ERROR] <ParseEdge> Edge does not match the format, para:', filename, line)
        return [], 1 # err output.
    content = contents[0]  # get the first match.

    # parse the content.
    content = content[1:-1].replace(' ', '')  # remove () and SPACE.
    segs = content.split(',')  # split with comma.
    segs[2] = ','.join(segs[2:-1])
    segs[2] = segs[2][1:-1]  # remove quotation marks.
    if segs[2].startswith('DDG'):
        segs[2] = 'DDG'
    if not segs[2] in ['CDG', 'DDG']:
        if _ERROR_: print('[ERROR] <ParseEdge> Edgetype Error, para:', filename, line)
        return [], 1 # err.

    # [nodeout, nodein, edgetype, version]
    ret = np.array([segs[0], segs[1], segs[2], segs[-1]], dtype=object)

    return ret, 0 # ok.

def ParseNode(filename, line):
    if _DEBUG_: print(line, end='')

    # the graph has no node.
    if '\n' == line:
        return [], 0 # ok.

    # find the pattern-matched content.
    pattern = r'\(-\d+, -?\d, \'[CD-]+\', \d+, \'[-+]?\d+\', [\'\"].*[\'\"]\)'
    contents = re.findall(pattern, line)
    if 0 == len(contents):
        if _ERROR_: print('[ERROR] <ParseNode> Node does not match the format, para:', filename, line)
        return [], 1 # err
    content = contents[0]

    # parse the content.
    content = content[1:-1]  # remove ()
    segs = content.split(',')  # split with comma.
    for i in range(0, 5):
        segs[i] = segs[i].replace(' ', '') # remove SPACE
    segs[2] = segs[2][1:-1]  # remove ''
    segs[4] = segs[4][1:-1]  # remove ''
    # parse statement.
    code = ','.join(segs[5:])
    while code[0] == ' ' and len(code) > 1:
        code = code[1:]
    while code[-1] == ' ' and len(code) > 1:
        code = code[:-1]
    code = code[1:-1]  # remove ''

    # [nodeid, version, nodetype, dist, linenum, [tokentype], [tokens]]
    ret = np.array([segs[0], segs[1], segs[2], segs[3], segs[4], code], dtype=object)

    return ret, 0

def ReadFile(filename):

    # read lines from the file.
    print('[INFO] <ReadFile> Read data from:', filename)
    fp = open(filename, encoding='utf-8', errors='ignore')
    lines = fp.readlines()
    fp.close()
    if _DEBUG_: print(lines)

    # get the data from edge and node information.
    ret_err = 0
    signGraph = 0
    signEdge = 1
    edges = {0: [], 1: [], 2: []} # 0:PatchCPG, 1:PreCPG, 2:PostCPG
    nodes = {0: [], 1: [], 2: []} # 0:PatchCPG, 1:PreCPG, 2:PostCPG

    for line in lines:
        # for each line in this file.
        if line.startswith('---'):
            # exchange to the next graph (0:PatchCPG, 1:PreCPG, 2:PostCPG).
            signGraph += 1
            signEdge = 1 # go back to edge scanning.
        elif line.startswith('==='):
            # exchange to the node parsing of the same graph.
            signEdge = 0 # start node scanning.
        # if the line is not a breakline, it can be parsed.
        elif (1 == signEdge): 
            # Edge:
            edge, err = ParseEdge(filename, line)
            if err: ret_err = 1
            if 0 == len(edge): # the edge is void.
                continue
            # if edge is good, save edge based on the graph.
            try:
                edges[signGraph].append(edge)
            except:
                print('[ERROR] <ReadFile> Find abnormal graphs, para:', filename)
        elif (0 == signEdge):
            # Node:
            node, err = ParseNode(filename, line)
            if err: ret_err = 1
            if 0 == len(node): # the node is void
                continue
            # if node is good, save node based on the graph.
            try:
                nodes[signGraph].append(node)
            except:
                print('[ERROR] <ReadFile> Find abnormal graphs, para:', filename)
        elif _ERROR_:
            # Error:
            print('[ERROR] <ReadFile> Neither an edge or a node, para:', filename, line)

    print(f'[INFO] <ReadFile> Read PatchCPG (#node: {len(nodes[0])}, #edge: {len(edges[0])}), ', end='')
    print(f'PreCPG (#node: {len(nodes[1])}, #edge: {len(edges[1])}), ', end='')
    print(f'PostCPG (#node: {len(nodes[2])}, #edge: {len(edges[2])}).' + RunTime())
    if _DEBUG_:
        print(nodes)
        print(edges)

    return nodes, edges, ret_err 

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'parse_graphs.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # check folders.
    if not os.path.exists(mdatPath + '/negatives/'):
        os.makedirs(mdatPath + '/negatives/')
    if not os.path.exists(mdatPath + '/positives/'):
        os.makedirs(mdatPath + '/positives/')
    # main entrance.
    main()
