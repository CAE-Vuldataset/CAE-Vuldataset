import os
import sys
import torch
from torch_geometric import __version__ as tg_version
from torch_geometric.loader import DataLoader
from .libs.PatchCPG import GetDataset, GetTrainDataset
from .libs.utils import TrainTestSplit, OutputEval, SaveBestModel, EndEpochLoop
from .libs.nets.GCN import GCN, GCNTrain, GCNTest
from .libs.nets.PGCN import PGCN, PGCN_g, PGCN_h, PGCNTrain, PGCNTest
from .utils import Logger

def Train_PatchGNN(opt):
    logfile = f'patch_gnn_{opt.net}_lr{opt.lr}_b{opt.batch_size}.txt'
    sys.stdout = Logger(os.path.join(opt.log_path, logfile))
    print('[INFO] CUDA Version: ' + (torch.version.cuda if torch.version.cuda else 'None'))
    print('[INFO] PyTorch Version: ' + torch.__version__)
    print('[INFO] Pytorch-Geometric Version: ' + tg_version)

    # load the PatchCPG dataset.
    # dataTrain, _ = GetDataset(path=opt.np_path)     # get dataset from local dataset.
    print("####################################################################")
    dataTrain, _ = GetTrainDataset(path=opt.np_path, txt=f'{opt.dataset}/train')
    print(len(dataTrain))
    dataTest, _ = GetTrainDataset(path=opt.vnp_path, txt=f'{opt.dataset}/val')
    dataset = dataTrain + dataTest
    print('[INFO] =============================================================')
    print(f'[INFO] Data instance: {dataset[0]}')
    print(f'[INFO] Dimension of node features: {dataset[0].num_features}')
    print(f'[INFO] Number of nodes: {dataset[0].num_nodes}')
    print(f'[INFO] Number of edges: {dataset[0].num_edges}')
    print('[INFO] =============================================================')

    # divide train set and test set.
    # dataTrain, dataTest = TrainTestSplit(dataset, train_size=opt.train_rate, random_state=100)
    print(f'[INFO] Number of training graphs: {len(dataTrain)}')
    print(f'[INFO] Number of test graphs: {len(dataTest)}')
    print(f'[INFO] Size of mini batch: {opt.batch_size}')
    print('[INFO] =============================================================')
    # get the train dataloader and test dataloader.
    trainloader = DataLoader(dataTrain, batch_size=opt.batch_size, shuffle=False)
    testloader = DataLoader(dataTest, batch_size=opt.batch_size, shuffle=False)

    # demo for graph neural network.
    if ('GCN' == opt.net): demo_GCN(opt, trainloader, testloader, dim_features=dataset[0].num_features)
    elif (opt.net.startswith('PGCN')): demo_PGCN(opt, trainloader, testloader, dim_features=dataset[0].num_features)
    else: print(f'[ERROR] argument {opt.net} is invalid!')

    return 0

def Test_PatchGNN(opt):
    print('[INFO] CUDA Version: ' + (torch.version.cuda if torch.version.cuda else 'None'))
    print('[INFO] PyTorch Version: ' + torch.__version__)
    print('[INFO] Pytorch-Geometric Version: ' + tg_version)

    # get parameter.
    dataset, _ = GetTrainDataset(path=opt.tnp_path, txt=f'{opt.dataset}/test')
    dim_features=dataset[0].num_features

    # model loader.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ('GCN' == opt.net):
        model = GCN(num_node_features=dim_features)
    elif (opt.net.startswith('PGCN')):
        if ('PGCN' == opt.net): model = PGCN(num_node_features=dim_features)
        elif ('PGCN_g' == opt.net): model = PGCN_g(num_node_features=dim_features)
        elif ('PGCN_h' == opt.net): model = PGCN_h(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if os.path.exists(opt.model_path + f'/model_{opt.net}_lr{opt.lr}_b{opt.batch_size}.pth'):
        model.load_state_dict(torch.load(opt.model_path + f'/model_{opt.net}_lr{opt.lr}_b{opt.batch_size}.pth'))
    else:
        print(f'[ERROR] Cannot find the model {opt.model_path}/model_{opt.net}_lr{opt.lr}_b{opt.batch_size}.pth')
        return 1
    
    # evaluation with the best model.
    testloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    testAcc, testPred, testLabel = PGCNTest(model, testloader)
    OutputEval(testPred, testLabel, opt.net)

    demo_eval(opt, model)
    print(f'[INFO] Find test result in {opt.log_path}/TestResult_{opt.net}_lr{opt.lr}_b{opt.batch_size}.txt')
    
    return 0

def demo_GCN(opt, trainloader, testloader, dim_features):
    # define device, model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCN(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if not (opt.use_model and os.path.exists(opt.model_path + f'/model_GCN_lr{opt.lr}_b{opt.batch_size}.pth')):
        # define optimizer, criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        criterion = torch.nn.CrossEntropyLoss()
        print(f'[INFO] Optimizer settings:\n{optimizer}')
        print(f'[INFO] Criterion settings: {criterion}')
        print(f'[INFO] Maximum epoch number: {opt.max_epoch}')
        print('[INFO] =============================================================')
        # train model.
        accList = [0] # accuracy recorder.
        for epoch in range(1, opt.max_epoch + 1):
            # train model and evaluate model.
            model, loss = GCNTrain(model, trainloader, optimizer=optimizer, criterion=criterion)
            trainAcc, trainPred, trainLabel = GCNTest(model, trainloader)
            testAcc, testPred, testLabel = GCNTest(model, testloader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f}')
            # save the best model.
            accList.append(testAcc)
            SaveBestModel(accList, model, path=opt.model_path, modelname='GCN', para=f'lr{opt.lr}_b{opt.batch_size}')
            # termination judgement.
            if (EndEpochLoop(accList, window=opt.win_size, firstepoch=opt.first_epoch)): break

    # evaluation with the best model.
    model.load_state_dict(torch.load(opt.model_path + f'/model_GCN_lr{opt.lr}_b{opt.batch_size}.pth'))
    testAcc, testPred, testLabel = GCNTest(model, testloader)
    OutputEval(testPred, testLabel, 'GCN')

    demo_eval(opt, model)

    return model

def demo_PGCN(opt, trainloader, testloader, dim_features):
    # define device, model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ('PGCN' == opt.net): model = PGCN(num_node_features=dim_features)
    elif ('PGCN_g' == opt.net): model = PGCN_g(num_node_features=dim_features)
    elif ('PGCN_h' == opt.net): model = PGCN_h(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if not (opt.use_model and os.path.exists(opt.model_path + f'/model_{opt.net}_lr{opt.lr}_b{opt.batch_size}.pth')):
        # define optimizer, criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        criterion = torch.nn.CrossEntropyLoss()
        print(f'[INFO] Optimizer settings:\n{optimizer}')
        print(f'[INFO] Criterion settings: {criterion}')
        print(f'[INFO] Maximum epoch number: {opt.max_epoch}')
        print('[INFO] =============================================================')
        # train model.
        accList = [0] # accuracy recorder.
        for epoch in range(1, opt.max_epoch + 1):
            # train model and evaluate model.
            model, loss = PGCNTrain(model, trainloader, optimizer=optimizer, criterion=criterion)
            trainAcc, trainPred, trainLabel = PGCNTest(model, trainloader)
            testAcc, testPred, testLabel = PGCNTest(model, testloader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f}')
            # save the best model.
            accList.append(testAcc)
            SaveBestModel(accList, model, path=opt.model_path, modelname=opt.net, para=f'lr{opt.lr}_b{opt.batch_size}')
            # termination judgement.
            if (EndEpochLoop(accList, window=opt.win_size, firstepoch=opt.first_epoch)): break

    # evaluation with the best model.
    model.load_state_dict(torch.load(opt.model_path + f'/model_{opt.net}_lr{opt.lr}_b{opt.batch_size}.pth'))
    testAcc, testPred, testLabel = PGCNTest(model, testloader)
    OutputEval(testPred, testLabel, opt.net)

    demo_eval(opt, model)

    return model

def demo_eval(opt, model):
    dataset, files = GetTrainDataset(path=opt.np_path, txt=f'{opt.dataset}/train') if opt.task == 'train' else GetTrainDataset(path=opt.tnp_path, txt=f'{opt.dataset}/test')
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    testAcc, testPred, testLabel = PGCNTest(model, dataloader)
    print(f'[INFO] The accuracy on all {opt.task} data is {testAcc*100:.3f}%.')

    if opt.task == 'train':
        filename = opt.log_path + f'/TrainResult_{opt.net}_lr{opt.lr}_b{opt.batch_size}.txt'
    else:
        filename = opt.log_path + f'/TestResult_{opt.net}_lr{opt.lr}_b{opt.batch_size}.txt'
    fp = open(filename, 'w')
    fp.write(f'filename,label,prediction\n')
    for i in range(len(files)):
        fp.write(f'{files[i][:-4]},{testLabel[i]},{testPred[i]}\n')
    fp.close()

    return testAcc
