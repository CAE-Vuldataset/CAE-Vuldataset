'''
    Patch-GNN
    Author:
        Shu Wang
    Dependencies:
        (For example, PyTorch 1.7.0 + CUDA 10.2)
        pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
        pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        pip install torch-geometric
'''

import os
import sys
import time
import argparse
import torch
from torch_geometric import __version__ as tg_version
from torch_geometric.data import DataLoader
from libs.PatchCPG_twin import PatchCPGDataset, GetDataset
from libs.utils import TrainTestSplit, OutputEval, SaveBestModel, EndEpochLoop
from libs.nets.GCN_twin import GCN_twin, GCNTrain, GCNTest
from libs.nets.PGCN_twin import PGCN_twin, PGCN_g_twin, PGCN_h_twin, PGCNTrain, PGCNTest

# environment settings.
rootPath = './'
tempPath = './'
dataPath = rootPath + '/_data/data_np2/'
pdatPath = rootPath + '/_data/PatchCPG/'
logsPath = tempPath + '/logs/'
mdlsPath = tempPath + '/models/'
# output parameters.
_DEBUG_  = 0    # 0: hide debug info. 1: show debug info.
_LOCAL_  = 1    # 0: use public data. 1: use local data.
_MODEL_  = 0    # 0: train new model. 1: use saved model.
# hyper-parameters.
_NETXARCHT_ = 'PGCN'
_BATCHSIZE_ = 128
_MAXEPOCHS_ = 400
_LEARNRATE_ = 0.01
_TRAINRATE_ = 0.8
_WINDOWSIZ_ = 0
_FISTEPOCH_ = 0
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
    # load the PatchCPG dataset.
    if _LOCAL_: dataset, _ = GetDataset(path=dataPath)     # get dataset from local dataset.
    else: dataset = PatchCPGDataset(root=pdatPath)      # get dataset from public dataset.
    print('[INFO] =============================================================')
    # print(f'[INFO] Load PatchCPG dataset: {dataset}')
    print(f'[INFO] Data instance: {dataset[0]}')
    print(f'[INFO] Dimension of node features: {len(dataset[0].x_s[0])}')
    print(f'[INFO] Number of nodes [1st]: {len(dataset[0].x_s)}')
    print(f'[INFO] Number of edges [1st]: {len(dataset[0].edge_attr_s)}')
    print(f'[INFO] Number of nodes [2nd]: {len(dataset[0].x_t)}')
    print(f'[INFO] Number of edges [2nd]: {len(dataset[0].edge_attr_t)}')
    print('[INFO] =============================================================')

    # divide train set and test set.
    dataTrain, dataTest = TrainTestSplit(dataset, train_size=_TRAINRATE_, random_state=100)
    print(f'[INFO] Number of training graphs: {len(dataTrain)}')
    print(f'[INFO] Number of test graphs: {len(dataTest)}')
    print(f'[INFO] Size of mini batch: {_BATCHSIZE_}')
    print('[INFO] =============================================================')
    # get the train dataloader and test dataloader.
    trainloader = DataLoader(dataTrain, batch_size=_BATCHSIZE_, follow_batch=['x_s', 'x_t'], shuffle=False)
    testloader = DataLoader(dataTest, batch_size=_BATCHSIZE_, follow_batch=['x_s', 'x_t'], shuffle=False)

    # demo for graph neural network.
    if ('GCN' == _NETXARCHT_): demo_GCN(trainloader, testloader, dim_features=len(dataset[0].x_s[0]))
    elif (_NETXARCHT_.startswith('PGCN')): demo_PGCN(trainloader, testloader, dim_features=len(dataset[0].x_s[0]))
    else: print('[ERROR] argument \'_NETXARCHT_\' is invalid!')

    return 0

def demo_GCN(trainloader, testloader, dim_features):
    # define device, model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCN_twin(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if not (_MODEL_ and os.path.exists(mdlsPath + f'/model_GCN_twin_{dim_features}.pth')):
        # define optimizer, criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNRATE_)
        criterion = torch.nn.CrossEntropyLoss()
        print(f'[INFO] Optimizer settings:\n{optimizer}')
        print(f'[INFO] Criterion settings: {criterion}')
        print(f'[INFO] Maximum epoch number: {_MAXEPOCHS_}')
        print('[INFO] =============================================================')
        # train model.
        accList = [0] # accuracy recorder.
        for epoch in range(1, _MAXEPOCHS_ + 1):
            # train model and evaluate model.
            model, loss = GCNTrain(model, trainloader, optimizer=optimizer, criterion=criterion)
            trainAcc, trainPred, trainLabel = GCNTest(model, trainloader)
            testAcc, testPred, testLabel = GCNTest(model, testloader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f}')
            # save the best model.
            accList.append(testAcc)
            SaveBestModel(accList, model, path=mdlsPath, modelname='GCN_twin', para=dim_features)
            # termination judgement.
            if (EndEpochLoop(accList, window=_WINDOWSIZ_, firstepoch=_FISTEPOCH_)): break

    # evaluation with the best model.
    model.load_state_dict(torch.load(mdlsPath + f'/model_GCN_twin_{dim_features}.pth'))
    testAcc, testPred, testLabel = GCNTest(model, testloader)
    OutputEval(testPred, testLabel, 'GCN_twin')

    return model

def demo_PGCN(trainloader, testloader, dim_features):
    # define device, model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ('PGCN' == _NETXARCHT_): model = PGCN_twin(num_node_features=dim_features)
    elif ('PGCN_g' == _NETXARCHT_): model = PGCN_g_twin(num_node_features=dim_features)
    elif ('PGCN_h' == _NETXARCHT_): model = PGCN_h_twin(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if not (_MODEL_ and os.path.exists(mdlsPath + f'/model_{_NETXARCHT_}_twin_{dim_features}.pth')):
        # define optimizer, criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNRATE_)
        criterion = torch.nn.CrossEntropyLoss()
        print(f'[INFO] Optimizer settings:\n{optimizer}')
        print(f'[INFO] Criterion settings: {criterion}')
        print(f'[INFO] Maximum epoch number: {_MAXEPOCHS_}')
        print('[INFO] =============================================================')
        # train model.
        accList = [0] # accuracy recorder.
        for epoch in range(1, _MAXEPOCHS_ + 1):
            # train model and evaluate model.
            model, loss = PGCNTrain(model, trainloader, optimizer=optimizer, criterion=criterion)
            trainAcc, trainPred, trainLabel = PGCNTest(model, trainloader)
            testAcc, testPred, testLabel = PGCNTest(model, testloader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f}')
            # save the best model.
            accList.append(testAcc)
            SaveBestModel(accList, model, path=mdlsPath, modelname=_NETXARCHT_+'_twin', para=dim_features)
            # termination judgement.
            if (EndEpochLoop(accList, window=_WINDOWSIZ_, firstepoch=_FISTEPOCH_)): break

    # evaluation with the best model.
    model.load_state_dict(torch.load(mdlsPath + f'/model_{_NETXARCHT_}_twin_{dim_features}.pth'))
    testAcc, testPred, testLabel = PGCNTest(model, testloader)
    OutputEval(testPred, testLabel, _NETXARCHT_+'_twin')

    return model

def ArgsParser():
    # define argument parser.
    parser = argparse.ArgumentParser()
    # add arguments.
    parser.add_argument('-net', help='the type of graph neural network', choices=['GCN', 'PGCN', 'PGCN_g', 'PGCN_h'])
    parser.add_argument('-batch', metavar='BATCHSIZE', help='the batch size of data loader', type=int)
    parser.add_argument('-epoch', metavar='MAXEPOCH', help='the max number of epochs', type=int)
    parser.add_argument('-lr', metavar='LEARNRATE', help='the learning rate for training', type=float)
    parser.add_argument('-tr', metavar='TRAINRATE', help='the rate of training set', type=float)
    parser.add_argument('-m', '--usemodel', help='use the existing saved model', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', '--local', help='use local dataset', action='store_true')
    group.add_argument('-p', '--public', help='use public dataset', action='store_true')
    parser.add_argument('-path', metavar='DATAPATH', help='the path to store dataset')
    parser.add_argument('-mp', metavar='MODELPATH', help='the path to store model')
    parser.add_argument('-lp', metavar='LOGSPATH', help='the path to store log file')
    parser.add_argument('-w', metavar='WINDOWSIZE', help='the window to judge loop end', type=int)
    parser.add_argument('-f', metavar='FIRSTEPOCH', help='the first epoch to judge loop end', type=int)
    # parse the arguments.
    args = parser.parse_args()
    # global variables.
    global _NETXARCHT_
    global _BATCHSIZE_
    global _MAXEPOCHS_
    global _LEARNRATE_
    global _TRAINRATE_
    global _WINDOWSIZ_
    global _FISTEPOCH_
    global _MODEL_
    global _LOCAL_
    global dataPath
    global pdatPath
    global mdlsPath
    global logsPath
    # perform actions.
    if (args.net): _NETXARCHT_ = args.net
    if (args.batch): _BATCHSIZE_ = args.batch
    if (args.epoch): _MAXEPOCHS_ = args.epoch
    if (args.lr): _LEARNRATE_ = args.lr
    if (args.tr): _TRAINRATE_ = args.tr
    if (args.usemodel): _MODEL_ = 1
    if (args.local): _LOCAL_ = 1
    if (args.public): _LOCAL_ = 0
    if (args.path):
        if _LOCAL_: dataPath = args.path
        else: pdatPath = args.path
    if (args.mp): mdlsPath = args.mp
    if (args.lp): logsPath = args.lp
    if (args.w): _WINDOWSIZ_ = args.w
    if (args.f): _FISTEPOCH_ = args.f

    return parser

if __name__ == '__main__':
    # sparse the arguments.
    ArgsParser()
    # initialize the log file.
    logfile = f'patch_gnn_{_NETXARCHT_}_twin_b{_BATCHSIZE_}_lr{_LEARNRATE_}.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # set torch environment.
    print('[INFO] CUDA Version: ' + (torch.version.cuda if torch.version.cuda else 'None'))
    print('[INFO] PyTorch Version: ' + torch.__version__)
    print('[INFO] Pytorch-Geometric Version: ' + tg_version)
    # main entrance.
    main()
