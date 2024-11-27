import os
import sys
import torch
from torch_geometric import __version__ as tg_version
from torch_geometric.loader import DataLoader
from .libs.PatchCPG_twin import GetDataset
from .libs.utils import TrainTestSplit, OutputEval, SaveBestModel, EndEpochLoop
from .libs.nets.GCN_twin import GCN_twin, GCNTrain, GCNTest
from .libs.nets.PGCN_twin import PGCN_twin, PGCN_g_twin, PGCN_h_twin, PGCNTrain, PGCNTest
from .utils import Logger

def Train_TwinGNN(opt):
    logfile = f'twin_gnn_{opt.net}_lr{opt.lr}_b{opt.batch_size}.txt'
    sys.stdout = Logger(os.path.join(opt.log_path, logfile))
    print('[INFO] CUDA Version: ' + (torch.version.cuda if torch.version.cuda else 'None'))
    print('[INFO] PyTorch Version: ' + torch.__version__)
    print('[INFO] Pytorch-Geometric Version: ' + tg_version)

    # load the PatchCPG dataset.
    dataset, _ = GetDataset(path=opt.np2_path)     # get dataset from local dataset.
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
    dataTrain, dataTest = TrainTestSplit(dataset, train_size=opt.train_rate, random_state=100)
    print(f'[INFO] Number of training graphs: {len(dataTrain)}')
    print(f'[INFO] Number of test graphs: {len(dataTest)}')
    print(f'[INFO] Size of mini batch: {opt.batch_size}')
    print('[INFO] =============================================================')
    # get the train dataloader and test dataloader.
    trainloader = DataLoader(dataTrain, batch_size=opt.batch_size, follow_batch=['x_s', 'x_t'], shuffle=False)
    testloader = DataLoader(dataTest, batch_size=opt.batch_size, follow_batch=['x_s', 'x_t'], shuffle=False)

    # demo for graph neural network.
    if ('GCN' == opt.net): demo_GCN(opt, trainloader, testloader, dim_features=len(dataset[0].x_s[0]))
    elif (opt.net.startswith('PGCN')): demo_PGCN(opt, trainloader, testloader, dim_features=len(dataset[0].x_s[0]))
    else: print(f'[ERROR] argument {opt.net} is invalid!')

    return 0

def Test_TwinGNN(opt):
    print('[INFO] CUDA Version: ' + (torch.version.cuda if torch.version.cuda else 'None'))
    print('[INFO] PyTorch Version: ' + torch.__version__)
    print('[INFO] Pytorch-Geometric Version: ' + tg_version)

    # get parameter.
    dataset, _ = GetDataset(path=opt.tnp2_path)
    dim_features=len(dataset[0].x_s[0])

    # model loader.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ('GCN' == opt.net):
        model = GCN_twin(num_node_features=dim_features)
    elif (opt.net.startswith('PGCN')):
        if ('PGCN' == opt.net): model = PGCN_twin(num_node_features=dim_features)
        elif ('PGCN_g' == opt.net): model = PGCN_g_twin(num_node_features=dim_features)
        elif ('PGCN_h' == opt.net): model = PGCN_h_twin(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if os.path.exists(opt.model_path + f'/model_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.pth'):
        model.load_state_dict(torch.load(opt.model_path + f'/model_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.pth'))
    else:
        print(f'[ERROR] Cannot find the model {opt.model_path}/model_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.pth')
        return 1
    
    demo_eval(opt, model)
    print(f'[INFO] Find test result in {opt.log_path}/TestResult_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.txt')

    return 0

def demo_GCN(opt, trainloader, testloader, dim_features):
    # define device, model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCN_twin(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if not (opt.use_model and os.path.exists(opt.model_path + f'/model_GCN_twin_lr{opt.lr}_b{opt.batch_size}.pth')):
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
            SaveBestModel(accList, model, path=opt.model_path, modelname='GCN_twin', para=f'lr{opt.lr}_b{opt.batch_size}')
            # termination judgement.
            if (EndEpochLoop(accList, window=opt.win_size, firstepoch=opt.first_epoch)): break

    # evaluation with the best model.
    model.load_state_dict(torch.load(opt.model_path + f'/model_GCN_twin_lr{opt.lr}_b{opt.batch_size}.pth'))
    testAcc, testPred, testLabel = GCNTest(model, testloader)
    OutputEval(testPred, testLabel, 'GCN_twin')

    demo_eval(opt, model)

    return model

def demo_PGCN(opt, trainloader, testloader, dim_features):
    # define device, model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ('PGCN' == opt.net): model = PGCN_twin(num_node_features=dim_features)
    elif ('PGCN_g' == opt.net): model = PGCN_g_twin(num_node_features=dim_features)
    elif ('PGCN_h' == opt.net): model = PGCN_h_twin(num_node_features=dim_features)
    print(f'[INFO] Running device: {device}')
    print(f'[INFO] Model definition:\n{model}')

    if not (opt.use_model and os.path.exists(opt.model_path + f'/model_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.pth')):
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
            SaveBestModel(accList, model, path=opt.model_path, modelname=opt.net+'_twin', para=f'lr{opt.lr}_b{opt.batch_size}')
            # termination judgement.
            if (EndEpochLoop(accList, window=opt.win_size, firstepoch=opt.first_epoch)): break

    # evaluation with the best model.
    model.load_state_dict(torch.load(opt.model_path + f'/model_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.pth'))
    testAcc, testPred, testLabel = PGCNTest(model, testloader)
    OutputEval(testPred, testLabel, opt.net+'_twin')

    demo_eval(opt, model)

    return model

def demo_eval(opt, model):
    dataset, files = GetDataset(path=opt.np2_path) if opt.task == 'train' else GetDataset(path=opt.tnp2_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, follow_batch=['x_s', 'x_t'], shuffle=False)
    testAcc, testPred, testLabel = PGCNTest(model, dataloader)
    print(f'[INFO] The accuracy on all {opt.task} data is {testAcc*100:.3f}%.')

    if opt.task == 'train':
        filename = opt.log_path + f'/TrainResult_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.txt'
    else:
        filename = opt.log_path + f'/TestResult_{opt.net}_twin_lr{opt.lr}_b{opt.batch_size}.txt'
    fp = open(filename, 'w')
    fp.write(f'filename,label,prediction\n')
    for i in range(len(files)):
        fp.write(f'{files[i][:-4]},{testLabel[i]},{testPred[i]}\n')
    fp.close()

    return testAcc
