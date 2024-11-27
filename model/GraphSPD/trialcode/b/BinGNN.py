import os
import sys
import time
import argparse
import numpy as np
import torch
from torch_geometric import __version__ as tg_version
from torch_geometric.loader import DataLoader
from libs.dataset import GetDataset
# from libs.graphvis import VisualGraph, VisualGraphs
from libs.utils import TrainTestSplit, OutputEval, SaveBestModel, EndEpochLoop
from libs.nets.BGNN_sliced import BGNN, BGNNTrain, BGNNTest

# environment settings.
rootPath = './'
tempPath = './'
dataPath = rootPath + '/data_np2/'
logsPath = tempPath + '/logs/'
mdlsPath = tempPath + '/models/'
# output parameters.
_DEBUG_  = 0    # 0: hide debug info. 1: show debug info.
_MODEL_  = 0    # 0: train new model. 1: use saved model.
# hyper-parameters.
_BATCHSIZE_ = 128
_MAXEPOCHS_ = 1000
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
    dataset, filelist = GetDataset(path=dataPath)  # get dataset from local dataset.
    # VisualGraphs(dataset[0], state=1)

    # divide train set and test set.
    dataTrain, dataTest = TrainTestSplit(dataset, train_size=_TRAINRATE_, allow_shuffle=False)
    fileTrain, fileTest = filelist[0:len(dataTrain)], filelist[len(dataTrain):]
    print(f'[INFO] Number of training graphs: {len(dataTrain)}')
    print(f'[INFO] Number of test graphs: {len(dataTest)}')
    print(f'[INFO] Size of mini batch: {_BATCHSIZE_}')
    print('[INFO] =============================================================')
    # get the train dataloader and test dataloader.
    trainloader = DataLoader(dataTrain, batch_size=_BATCHSIZE_, follow_batch=['x_s', 'x_t'], shuffle=False)
    testloader = DataLoader(dataTest, batch_size=_BATCHSIZE_, follow_batch=['x_s', 'x_t'], shuffle=False)

    # demo for graph neural network.
    model = demo_BinGNN(trainloader, testloader, dim_features=len(dataset[0].x_s[0]))
    # output results on test data.
    _, testPred, _ = BGNNTest(model, testloader)
    with open(logsPath + 'TestResults.txt', 'w') as f:
        for i in range(len(fileTest)):
            f.write(fileTest[i] + ',' + str(testPred[i]) + '\n')

    return

def demo_BinGNN(trainloader, testloader, dim_features):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BGNN(num_node_features=dim_features)

    if not (_MODEL_ and os.path.exists(mdlsPath + f'/model_BGNN_{dim_features}.pth')):
        # define optimizer, criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNRATE_)
        criterion = torch.nn.CrossEntropyLoss()
        print(f'[INFO] Optimizer settings:\n{optimizer}')
        print(f'[INFO] Criterion settings: {criterion}')
        print(f'[INFO] Maximum epoch number: {_MAXEPOCHS_}')
        print('[INFO] =============================================================')
        # train model.
        accList = [0]  # accuracy recorder.
        for epoch in range(1, _MAXEPOCHS_ + 1):
            # train model and evaluate model.
            model, loss = BGNNTrain(model, trainloader, optimizer=optimizer, criterion=criterion)
            trainAcc, trainPred, trainLabel = BGNNTest(model, trainloader)
            testAcc, testPred, testLabel = BGNNTest(model, testloader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f}')
            # save the best model.
            accList.append(testAcc)
            SaveBestModel(accList, model, path=mdlsPath, modelname='BGNN', para=dim_features)
            # termination judgement.
            if (EndEpochLoop(accList, window=_WINDOWSIZ_, firstepoch=_FISTEPOCH_)): break

    # evaluation with the best model.
    model.load_state_dict(torch.load(mdlsPath + f'/model_BGNN_{dim_features}.pth'))
    testAcc, testPred, testLabel = BGNNTest(model, testloader)
    OutputEval(testPred, testLabel, 'BGNN')

    return model

if __name__ == '__main__':
    # sparse the arguments.
    # initialize the log file.
    logfile = f'BinGNN.txt'
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
    main()