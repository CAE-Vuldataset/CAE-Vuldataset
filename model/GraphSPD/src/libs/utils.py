import os
import random
import numpy as np
import torch

def TrainTestSplit(dataset, train_size=0.8, allow_shuffle=True, random_state=0):
    '''
    Split the training set and testing set.
    :param dataset: incoming dataset
    :param train_size: the rate of the training set to the total set.
    :param allow_shffle: identifier if it is allowed to shuffle the dataset.
    :param random_state: the random state.
    :return: dataTrain, dataTest
    '''

    # shuffle the dataset.
    if (allow_shuffle):
        if (type(dataset) == list):
            random.Random(random_state).shuffle(dataset)
        else:
            torch.manual_seed(random_state)
            dataset = dataset.shuffle()

    # get the number of train set.
    numTrain = int(len(dataset) * train_size)

    # divide train set and test set.
    dataTrain = dataset[:numTrain]
    dataTest = dataset[numTrain:]

    return dataTrain, dataTest

def Evaluation(predictions, labels):
    '''
    Evaluate the predictions with gold labels, and get accuracy and confusion matrix.
    :param predictions: [0, 1, 0, ...]
    :param labels: [0, 1, 1, ...]
    :return: accuracy - 0~1
             confusion - [[1000, 23], [12, 500]]
    '''

    # parameter settings.
    D = len(labels)
    cls = 2

    # get confusion matrix.
    confusion = np.zeros((cls, cls))
    for ind in range(D):
        nRow = int(predictions[ind])
        nCol = int(labels[ind])
        confusion[nRow][nCol] += 1

    # get accuracy.
    accuracy = 0
    for ind in range(cls):
        accuracy += confusion[ind][ind]
    accuracy /= D

    return accuracy, confusion

# def OutputEval(predictions, labels, method=''):
#     '''
#     Output the evaluation results.
#     :param predictions: predicted labels. [[0], [1], ...]
#     :param labels: ground truth labels. [[1], [1], ...]
#     :param method: method name. string
#     :return: accuracy - the total accuracy. numeric
#              confusion - confusion matrix [[1000, 23], [12, 500]]
#     '''

#     # get accuracy and confusion matrix.
#     accuracy, confusion = Evaluation(predictions, labels)
#     precision = confusion[1][1] / (confusion[1][0] + confusion[1][1]) if (confusion[1][0] + confusion[1][1]) else 0
#     recall = confusion[1][1] / (confusion[0][1] + confusion[1][1]) if (confusion[0][1] + confusion[1][1]) else 0
#     F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

#     # output on screen and to file.
#     print('       -------------------------------------------')
#     print('       method           :  ' +  method) if len(method) else print('', end='')
#     print('       accuracy  (ACC)  :  %.3f%%' % (accuracy * 100))
#     print('       precision (P)    :  %.3f%%' % (precision * 100))
#     print('       recall    (R)    :  %.3f%%' % (recall * 100))
#     print('       F1 score  (F1)   :  %.3f' % (F1))
#     print('       fall-out  (FPR)  :  %.3f%%' % (confusion[1][0] * 100 / (confusion[1][0] + confusion[0][0]) if (confusion[1][0] + confusion[0][0]) else 0))
#     print('       miss rate (FNR)  :  %.3f%%' % (confusion[0][1] * 100 / (confusion[0][1] + confusion[1][1]) if (confusion[0][1] + confusion[1][1]) else 0))
#     print('       confusion matrix :      (actual)')
#     print('                           Neg         Pos')
#     print('       (predicted) Neg     %-5d(TN)   %-5d(FN)' % (confusion[0][0], confusion[0][1]))
#     print('                   Pos     %-5d(FP)   %-5d(TP)' % (confusion[1][0], confusion[1][1]))
#     print('       -------------------------------------------')

#     return accuracy, confusion

def OutputEval(predictions, labels, method=''):
    '''
    Output the evaluation results.
    :param predictions: predicted labels. [[0], [1], ...]
    :param labels: ground truth labels. [[1], [1], ...]
    :param method: method name. string
    :return: accuracy - the total accuracy. numeric
             confusion - confusion matrix [[1000, 23], [12, 500]]
    '''

    # get accuracy and confusion matrix.
    accuracy, confusion = Evaluation(predictions, labels)
    TP = confusion[1][1]
    TN = confusion[0][0]
    FP = confusion[1][0]
    FN = confusion[0][1]

    precision = TP / (FP + TP) if (FP + TP) else 0
    recall = TP / (FN + TP) if (FN + TP) else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    # Calculate MCC
    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = (TP * TN - FP * FN) / denominator**0.5 if denominator else 0

    # output on screen and to file.
    print('       -------------------------------------------')
    print('       method           :  ' +  method) if len(method) else print('', end='')
    print('       accuracy  (ACC)  :  %.3f%%' % (accuracy * 100))
    print('       precision (P)    :  %.3f%%' % (precision * 100))
    print('       recall    (R)    :  %.3f%%' % (recall * 100))
    print('       F1 score  (F1)   :  %.3f' % (F1))
    print('       MCC              :  %.3f' % (MCC))
    print('       fall-out  (FPR)  :  %.3f%%' % (FP * 100 / (FP + TN) if (FP + TN) else 0))
    print('       miss rate (FNR)  :  %.3f%%' % (FN * 100 / (FN + TP) if (FN + TP) else 0))
    print('       confusion matrix :      (actual)')
    print('                           Neg         Pos')
    print('       (predicted) Neg     %-5d(TN)   %-5d(FN)' % (TN, FN))
    print('                   Pos     %-5d(FP)   %-5d(TP)' % (FP, TP))
    print('       -------------------------------------------')

    return accuracy, confusion


def SaveBestModel(accuracylist, model, path='./', modelname='NA', para='NA'):
    '''
    Save the model if the accuracy is the highest.
    :param accuracylist: accuracy list [0, 0.1, 0.23, ...]
    :param model: model variable.
    :param path: the path to store the model.
    :param modelname: the filename to store the model.
    :param para: the parameter to store the model.
    :return: if the model is saved.
    '''

    if (len(accuracylist) <= 1):
        print('[ERROR] <SaveBestModel> length of argument \'accuracyList\' is less than 2')
        return False

    if (accuracylist[-1] > max(accuracylist[:-1])):  # if the test accuracy is higher than previous accuracies.
        # check the saving folder.
        if not os.path.exists(path):
            os.makedirs(path)
        # save model.
        torch.save(model.state_dict(), path + f'/model_{modelname}_{para}.pth')
        return True

    return False

def EndEpochLoop(accuracylist, window=0, firstepoch=0):
    '''
    Judge if it is time to end the training loop.
    :param accuracylist: accuracy list [0, 0.1, 0.23, ...] with epoch 0, 1, 2, ...
    :param window: the judgement window size.
    :param firstepoch: the first epoch to judge.
    :return: if the training loop should be terminated.
    '''

    if (window <= 0): # run all epoch.
        return False

    if (len(accuracylist) <= firstepoch): # judge from the firstepoch.
        return False

    if (len(accuracylist) <= window): # check the window condition.
        return False

    if (accuracylist[-1] < min(accuracylist[-1 - window:-1])):
        return True

    return False
