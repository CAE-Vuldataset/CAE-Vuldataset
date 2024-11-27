import os
import re
import numpy as np
import scipy.sparse as sp
import json

def ReadFuncInfo(opt, commitID):
    abfsPath = opt.ab_path
    ## Read joern meta data and get the function info.
    funcInfo = [[], []] # [fileName, funcName, lineStart, lineEnd, cpgPath, funcChg, lineChg]

    for vidx, ver in enumerate(['A', 'B']):
        metafile = os.path.join(abfsPath, commitID+f'/func{ver}.txt')
        if os.path.exists(metafile):
            func = open(metafile, encoding='utf-8', errors='ignore').read()
            funcList = json.loads(func)

            for idx, item in enumerate(funcList):
                if 4 == len(item.keys()):
                    # funcInfo[vidx].append([item['_1'][item['_1'].find(f'/{ver.lower()}/'):], # fileName
                    funcInfo[vidx].append([f"/{ver.lower()}/{item['_1']}", # fileName
                                           "&lt;global&gt;" if '<global>' in item['_2'] else item['_2'], # funcName 
                                           item['_3'], item['_4'], f'cpgs{ver}/{idx}-cpg.dot', 0, []])
    return np.array(funcInfo, dtype=object)

def ReadHunkInfo(opt, commitID):
    abfsPath = opt.ab_path
    ## Read diff file and get the hunk info.
    hunkInfo = []

    # read the diff file.
    diffLines = open(os.path.join(abfsPath, commitID+'/diff.patch'), encoding='utf-8', errors='ignore').readlines()
    
    # go through the lines in diff file.
    pattern = r'@@ -(\d+),*(\d+)* \+(\d+),*(\d+)* @@.*'
    for line in diffLines:
        # determine the filename in A, B.
        if line.startswith("diff -brN -U 0 -p"):
            [fnameA, fnameB] = line.split()[-2:]
            fnameA = fnameA[fnameA.find('/a/'):]
            fnameB = fnameB[fnameB.find('/b/'):]

        # find @@ -XX,XX +XX,XX @@ line.
        contents = re.findall(pattern, line)
        if len(contents):
            lnums = [int(c) if len(c) else 1 for c in contents[0]]
            hunkInfo.append([fnameA, fnameB, 
                             lnums[0], lnums[0]+lnums[1]-1 if lnums[1] else 0,
                             lnums[2], lnums[2]+lnums[3]-1 if lnums[3] else 0])

    return np.array(hunkInfo, dtype=object)

def FindChgdFunc(funcInfo, hunkInfo):
    # find each pair of func and hunk.
    for hunk in hunkInfo:
        for func in funcInfo[0]: # A.
            # func.fileName == hunk.fileName
            if func[0] == hunk[0] and hunk[3] != 0: # has changed code.
                if (max(func[2], hunk[2]) <= min(func[3], hunk[3])): # has overlapping.
                    func[-2] = 1 if func[1] != "&lt;global&gt;" else 0 # function changed.
                    if func[1] != "&lt;global&gt;": func[-1].extend(range(hunk[2], hunk[3]+1)) # changed line.
            elif func[0] == hunk[0] and hunk[3] == 0: # no changed code.
                if func[2] <= hunk[2] and hunk[2] <= func[3]: 
                    func[-2] = 1 if func[1] != "&lt;global&gt;" else 0 # provide context.
            # else no matched filenames.
        for func in funcInfo[1]: # B.
            if func[0] == hunk[1] and hunk[5] != 0:
                if (max(func[2], hunk[4]) <= min(func[3], hunk[5])): # has overlapping.
                    func[-2] = 1 if func[1] != "&lt;global&gt;" else 0 # function changed.
                    if func[1] != "&lt;global&gt;": func[-1].extend(range(hunk[4], hunk[5]+1)) # changed line.
            elif func[0] == hunk[1] and hunk[5] == 0:
                if func[2] <= hunk[4] and hunk[4] <= func[3]: 
                    func[-2] = 1 if func[1] != "&lt;global&gt;" else 0 # provide context.
    
    # return
    chgdFuncInfo = [[], []]
    for v in range(2):
        for func in funcInfo[v]:
            if func[-2] == 1:
                chgdFuncInfo[v].append(func)

    return np.array(chgdFuncInfo, dtype=object)

def GetCodeNodes(opt, chgdFuncInfo, commitID):
    abfsPath = opt.ab_path
    nodes = [[], []]

    nid = 0
    for v in range(2):
        for func in chgdFuncInfo[v]:
            ndot = []
            ver = -1 if 0 == v else 1 # version info
            codeLines = open(os.path.join(abfsPath, commitID+func[0]), encoding='utf-8', errors='ignore').readlines() # func[0] -> read code lines 
            for lnum in range(func[2], func[3]+1): # func[2], func[3] -> involved func lines
                code = codeLines[lnum-1].strip()
                ndot.append([nid, lnum, lnum, ver if lnum in func[-1] else 0, code]) # func[-1] -> ver
                nid += 1
            # append the dot.
            nodes[v].append(np.array(ndot, dtype=object))

    return np.array(nodes, dtype=object)

def AlignedMerge(nodes, chgdFuncInfo, hunkInfo):

    def ReOrder(ndot, atLines, version=0):
        acc = [0, 0]
        for line in atLines:
            # recover diff file line.
            l = [line[2], max(0, line[3]-line[2]+1), line[4], max(0, line[5]-line[4]+1)]
            # get shift start point.
            if version == 0: # A
                if l[1] != 0: # a != 0
                    threshold = l[0] + l[1] + acc[0] # acc + A + a
                else: # a == 0
                    threshold = l[2] + acc[1] # acc + B
            elif version == 1: # B
                if l[3] != 0: # b != 0
                    threshold = l[2] + acc[1] # acc + B
                else: # b == 0
                    threshold = l[0] + acc[0] # acc + A
            # get shift value.
            diff = l[3] if version == 0 else l[1]
            # record the total shifts.
            acc[0] += l[3]
            acc[1] += l[1]
            # print(threshold, diff, acc)
            for node in ndot:
                if node[1] >= threshold:
                    node[1] += diff
        return ndot

    # re-order.
    for v in range(2): # go through A and B.
        for meta, ndot in zip(chgdFuncInfo[v], nodes[v]): # func_meta, func_dot
            iList = np.where(hunkInfo.T[v] == meta[0])[0] # find where the filename is the same.
            hInfo = hunkInfo[iList] # get the hunk info for the filename.
            ndot = ReOrder(ndot, hInfo, v) # reorder the line number. 

    # merge nodes.
    ret = [] 
    # load dictionary.
    filesAB = list(set([h[0]+' '+h[1] for h in hunkInfo]))
    dictAB = np.array([[ab.split()[0], ab.split()[1], dict()] for ab in filesAB]).T
    for meta, ndot in zip(chgdFuncInfo[0], nodes[0]):
        x = np.where(dictAB[0]==meta[0])[0][0] # find index of fileA.
        for n in ndot:
            dictAB[2][x][n[1]] = n[0] # linenum -> nodeID.
            ret.append(n) # get all nodes.

    # map dictionary.
    for meta, ndot in zip(chgdFuncInfo[1], nodes[1]):
        x = np.where(dictAB[1]==meta[0])[0][0] # find index of fileB.
        for n in ndot:
            if n[1] in dictAB[2][x].keys():
                n[0] = dictAB[2][x][n[1]] # map the nodeID.
            else:
                ret.append(n) # get additional nodes.

    # get the dictionary used to map the edges.
    dictE = np.array([[ab.split()[0], ab.split()[1], dict(), dict()] for ab in filesAB]).T
    for v in range(2):
        for meta, ndot in zip(chgdFuncInfo[v], nodes[v]):
            x = np.where(dictE[v]==meta[0])[0][0] # find index of file.
            for n in ndot:
                dictE[v+2][x][n[2]] = n[0] # orignal linenum -> nodeID.

    return np.array(ret, dtype=object), dictE

def GetCodeEdges(opt, chgdFuncInfo, commitID, nodes, dictE):

    def ImportCPG(dotfile, dictM):
        pattern_n = r"\"(\d+)\" \[label = <.*<SUB>(\d+)</SUB>> \]"
        pattern_e = r"  \"(\d+)\" -> \"(\d+)\"  \[ label = \"(.*)\"\] "
        # get the node dict(_nid_->linenum), node_list.
        dictN = {}
        if not os.path.exists(dotfile):
            return []
        lines = open(dotfile, encoding='utf-8', errors='ignore').readlines()
        for line in lines:
            contents = re.findall(pattern_n, line) # match nodes.
            if len(contents):
                (_nid_, _lnum_) = contents[0] 
                dictN[_nid_] = int(_lnum_)
        # get edges
        edges = []
        for line in lines:
            contents = re.findall(pattern_e, line) # match edges.
            if len(contents):
                (_nout_, _nin_, _etype_) = contents[0]
                if _nout_ not in dictN.keys() or _nin_ not in dictN.keys() or _etype_.startswith("AST"):
                    continue # filter out abnormal edges.
                if dictN[_nout_] not in dictM.keys() or dictN[_nin_] not in dictM.keys():
                    continue
                edges.append(f"{str(dictM[dictN[_nout_]])} {str(dictM[dictN[_nin_]])} {_etype_[:3]} 0")
        edges = list(set(edges))
        return edges

    abfsPath = opt.ab_path
    # load the edges strings.
    edges = []
    for v in range(2):
        for meta in chgdFuncInfo[v]:
            x = np.where(dictE[v]==meta[0])[0][0] # find index of file.
            e = ImportCPG(os.path.join(abfsPath, f"{commitID}/{meta[4]}"), dictE[v+2][x])
            edges.extend(e)
    edges = list(set(edges))

    # get the version node list.
    nodeList = [[], []]
    for n in nodes:
        if n[3] == -1:
            nodeList[0].append(n[0]) # pre-patch nodes.
        elif n[3] == 1:
            nodeList[1].append(n[0]) # post-patch nodes.
    
    # normalize the edges.
    ret = []
    for e in edges:
        [nout, nin, etype, ver] = e.split() # for each edge
        if nout == nin: # loop edge
            continue
        if int(nout) in nodeList[0] or int(nin) in nodeList[0]:
            ver = -1
        elif int(nout) in nodeList[1] or int(nin) in nodeList[1]:
            ver = 1
        if etype == "CFG":
            etype = "CDG"
        ret.append([int(nout), int(nin), etype, int(ver)])

    return np.array(ret, dtype=object)

def BriefNodes(nodes, edges):
    nodeList = np.r_[edges.T[0], edges.T[1]] if len(edges) else []
    _nodes_ = []
    for n in nodes:
        if n[0] in nodeList:
            _nodes_.append(n)
    return np.array(_nodes_, dtype=object)

def GraphSlicing(nodes, edges, neighbor=1):
    # get the mask.
    mask = np.array([1 if ver else 0 for ver in nodes.T[3]])
    # get node id dict.
    nodeIDs = dict((nID, idx) for idx, nID in enumerate(nodes.T[0]))
    # get the adjency matrix.
    nodesOut = [nodeIDs[n] for n in edges.T[0]]
    nodesIn = [nodeIDs[n] for n in edges.T[1]]
    A = sp.coo_matrix((np.ones(len(edges)), (nodesOut, nodesIn)), 
                      shape=(len(nodes), len(nodes)), dtype="float32")
    I = sp.eye(len(nodes)) # identity matrix.
    # calcualte total adjency matrix.
    M = I
    for h in range(1, neighbor+1):
        M += (A + A.T) ** h
    # calcualte the mask.
    _mask_ = np.array([1 if i else 0 for i in mask * M])

    return _mask_

def GraphWalk(nodes, edges, mask, step=1, direction='forward', contained=False):
    # get node id dict.
    nodeIDs = dict((nID, idx) for idx, nID in enumerate(nodes.T[0]))
    # get the adjency matrix.
    nodesOut = [nodeIDs[n] for n in edges.T[0]]
    nodesIn = [nodeIDs[n] for n in edges.T[1]]
    A = sp.coo_matrix((np.ones(len(edges)), (nodesOut, nodesIn)), 
                      shape=(len(nodes), len(nodes)), dtype="float32")
    I = sp.eye(len(nodes)) # identity matrix.
    # calcualte total adjency matrix.
    if direction == 'forward':
        M = A
    elif direction == 'backward':
        M = A.T
    elif direction == 'bidirect':
        M = A + A.T
    if contained:
        temp = I
        while (step):
            temp += M ** step
            step -= 1
        M = temp
    else:
        M = M ** step
    # calcualte the mask.
    print(M.toarray())
    _mask_ = np.array([1 if i else 0 for i in mask * M])

    return _mask_

def BriefGraph(nodes, edges, mask):
    # get the nodes.
    _nodes_ = []
    nodeList = []
    for msk, n in zip(mask, nodes):
        if msk:
            _nodes_.append(n)
            nodeList.append(n[0])
    _nodes_ = np.array(_nodes_, dtype=object)

    # get the edges.
    _edges_ = []
    for e in edges:
        if e[0] in nodeList and e[1] in nodeList:
            _edges_.append(e)
    _edges_ = np.array(_edges_, dtype=object)

    return _nodes_, _edges_

def ParseCPG14(opt, commitID):
    # get the function info.
    funcInfo = ReadFuncInfo(opt, commitID)
    # get the hunk info.
    hunkInfo = ReadHunkInfo(opt, commitID)
    # find the involved changed function info.
    chgdFuncInfo = FindChgdFunc(funcInfo, hunkInfo)
    # get the code nodes from involved functions, mark the changed lines.
    nodes = GetCodeNodes(opt, chgdFuncInfo, commitID)
    # align the AB nodes, get the dict(linenum->nodeID).
    nodes, dictE = AlignedMerge(nodes, chgdFuncInfo, hunkInfo)
    # get edges for dot files.
    edges = GetCodeEdges(opt, chgdFuncInfo, commitID, nodes, dictE)
    # remove the nodes not used in edges.
    nodes = BriefNodes(nodes, edges)
    if len(edges) == 0:
        return nodes, edges
    # code slicing.
    mask = GraphSlicing(nodes, edges, neighbor=opt.slicing)
    # remove the nodes not in mask and the inrelevant edges.
    nodes, edges = BriefGraph(nodes, edges, mask)

    return nodes, edges

def SplitGraphs(nodes, edges):
    # split nodes.
    _nodesA_, _nodesB_ = [], []
    for n in nodes:
        if n[3] <= 0:
            _nodesA_.append(n)
        if n[3] >= 0:
            _nodesB_.append(n)
    _nodesA_ = np.array(_nodesA_, dtype=object)
    _nodesB_ = np.array(_nodesB_, dtype=object)
    # split edges.
    _edgesA_, _edgesB_ = [], []
    for e in edges:
        if e[3] <= 0:
            _edgesA_.append(e)
        if e[3] >= 0:
            _edgesB_.append(e)
    _edgesA_ = np.array(_edgesA_, dtype=object)
    _edgesB_ = np.array(_edgesB_, dtype=object)

    return _nodesA_, _edgesA_, _nodesB_, _edgesB_
