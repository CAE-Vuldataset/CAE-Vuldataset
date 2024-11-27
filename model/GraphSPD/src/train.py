import os
import numpy as np
from src.get_cpg14 import GetFilesAB, GetCPG14
from src.parse_cpg14 import ParseCPG14, SplitGraphs
from src.embed_cpg14 import SetupConfigs, EmbedCPG14, EmbedCPG14Twin
from src.net_patch import Train_PatchGNN, Test_PatchGNN
from src.net_twin import Train_TwinGNN, Test_TwinGNN
from src.utils import *

def Data_Prep(opt):
    # Step 1.0: Get training data list.
    samples = open(opt.train_file).readlines() if opt.task == 'train' else open(opt.test_file).readlines()
    samples = [s.strip().split(',') for s in samples]

    # Step 1.1: Get relevent files in A/B and diff.patch. [db -> A | B | P]
    for idx, [owner, repo, commitID, _] in enumerate(samples):
        if not os.path.exists(os.path.join(opt.ab_path, commitID+'/diff.patch')):
            print(f'[INFO] [{idx+1}|{len(samples)}] Downloading the AB files for {owner}.{repo}.{commitID}{RunTime()}')
            GetFilesAB(opt, owner, repo, commitID)
        else:
            print(f'[INFO] [{idx+1}|{len(samples)}] Found downloaded AB files for {owner}.{repo}.{commitID}{RunTime()}')
    
    # Step 1.2: Get the subgraphs for the functions in A/B relevant files. [A -> Ag | Af; B -> Bg | Bf]
    commitList = os.listdir(opt.ab_path)
    samples = [s for s in samples if s[2] in commitList]
    for idx, [owner, repo, commitID, _] in enumerate(samples):
        if not os.path.exists(os.path.join(opt.ab_path, commitID+'/funcA.txt')) and \
           not os.path.exists(os.path.join(opt.ab_path, commitID+'/funcB.txt')): # funcA.txt and funcB.txt not exist simultaneously.
            print(f'[INFO] [{idx+1}|{len(samples)}] Building the joern subgraph files for {owner}.{repo}.{commitID}{RunTime()}')
            GetCPG14(opt, commitID)
        else:
            print(f'[INFO] [{idx+1}|{len(samples)}] Found the joern subgraph files for {owner}.{repo}.{commitID}{RunTime()}')

    # Step 2: Parse the subgraphs and build PatchCPGs. [Af, Bf, P, A, B, Ag, Bg -> mid]
    for idx, [_, _, commitID, label] in enumerate(samples):
        if not os.path.exists(os.path.join(opt.mid_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tmid_path, commitID+'.npz')):
            print(f'[INFO] [{idx+1}|{len(samples)}] Parsing the code property graph for {commitID}.{RunTime()}')
            # process the patch graph.
            nodes, edges = ParseCPG14(opt, commitID)
            if len(nodes) == 0 and len(edges) == 0: # void graph.
                continue # skip this samples.
            # process the twin graphs.
            if opt.twin_data:
                nodesA, edgesA, nodesB, edgesB = SplitGraphs(nodes, edges)
            else:
                emp = np.array([], dtype=object)
                nodesA, edgesA, nodesB, edgesB = emp, emp, emp, emp
            # save the mid-point graphs into the files.
            np.savez(os.path.join(opt.mid_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tmid_path, commitID+'.npz'), 
                     nodesP=nodes, edgesP=edges, nodesA=nodesA, edgesA=edgesA, 
                     nodesB=nodesB, edgesB=edgesB, label=[int(label)], dtype=object)
        else:
            print(f'[INFO] [{idx+1}|{len(samples)}] Found parsed code property graph for {commitID}.{RunTime()}')

    # Step 3: Embed the subgraphs into the numeric graphs.
    commitList = os.listdir(opt.mid_path) if opt.task == 'train' else os.listdir(opt.tmid_path)
    samples = [s for s in samples if s[2]+'.npz' in commitList]
    SetupConfigs(opt)
    for idx, [owner, repo, commitID, _] in enumerate(samples):
        # embed the patch graph.
        print('--------------------------------------------------------')
        if not os.path.exists(os.path.join(opt.np_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tnp_path, commitID+'.npz')):
            err = EmbedCPG14(opt, commitID)
            if err:
                print(f'[ERROR] [{idx+1}|{len(samples)}] There are error(s) in embedding (patch) CPG: {owner}.{repo}.{commitID}{RunTime()}')
            else:
                print(f'[INFO] [{idx+1}|{len(samples)}] Embedded (patch) CPG into numpy file {owner}.{repo}.{commitID}{RunTime()}')
        else:
            print(f'[INFO] [{idx+1}|{len(samples)}] Found embedded (patch) CPG file {owner}.{repo}.{commitID}{RunTime()}')
        # embed the twin graphs.
        if not os.path.exists(os.path.join(opt.np2_path, commitID+'.npz') if opt.task == 'train' else os.path.join(opt.tnp2_path, commitID+'.npz')) and opt.twin_data:
            err = EmbedCPG14Twin(opt, commitID)
            if err:
                print(f'[ERROR] [{idx+1}|{len(samples)}] There are error(s) in embedding (twin) CPG: {owner}.{repo}.{commitID}{RunTime()}')
            else:
                print(f'[INFO] [{idx+1}|{len(samples)}] Embedded (twin) CPG into numpy file {owner}.{repo}.{commitID}{RunTime()}')
        else:
            print(f'[INFO] [{idx+1}|{len(samples)}] Found embedded (twin) CPG file {owner}.{repo}.{commitID}{RunTime()}')

    print(f'[INFO] Finished preparing {opt.task}ing dataset!{RunTime()}')

    return 0

def Train(opt):
    if opt.twin:
        Train_TwinGNN(opt)
    else:
        Train_PatchGNN(opt)
    return 0

def Test(opt):
    if opt.twin:
        Test_TwinGNN(opt)
    else:
        Test_PatchGNN(opt)
    return 0