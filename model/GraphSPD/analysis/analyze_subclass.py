import os
import sys
import shutil
import random

rootPath = './'
tempPath = './'
dataPath = rootPath + '/_PatchDB/'
ndatPath = tempPath + '/data_np0/'
testPath = tempPath + '/data_np/'
logsPath = tempPath + '/logs/'

samp_ratio = 21640 / 8888

# Logger: redirect the stream on screen and to file.
# class Logger(object):
#     def __init__(self, filename = "log.txt"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#     def flush(self):
#         pass

def main():
    numClass = len(os.listdir(os.path.join(dataPath, 'positives')))

    for n in range(numClass):
        print('====================================================================')
        # get neg and pos file list.
        _, posLists = Get_FileList(dataPath)
        # set block list.
        posClass = list(posLists.keys())
        blockCls = ['OOB_access', 'lock', 'complicated', 'uninitialized_use', posClass[n]]
        blockCls = list(set(blockCls))
        print(f'[INFO] Sub-Class Block List: {blockCls}')
        # get the remaining pos list.
        for i in range(len(blockCls)):
            posLists.pop(blockCls[i])
        posList = [file for lst in posLists.values() for file in lst]
        posList = list(set(posList))

        # file operations [!important]
        # remove test folder.
        if (os.path.exists(testPath)):
            shutil.rmtree(testPath)
        # copy pos samples.
        os.makedirs(testPath + '/positives/')
        posNum = 0
        for file in os.listdir(ndatPath + '/positives/'):
            if file[:-4] in posList:
                posNum += 1
                shutil.copy(ndatPath + '/positives/' + file, testPath + '/positives/')

        # calculate the number of neg samples.
        negNum = int(posNum * samp_ratio)
        # get the corresponding neg list.
        negList = os.listdir(ndatPath + '/negatives/')
        negList = random.sample(negList, negNum)
        

        # file operations [!important]
        # copy neg samples.
        os.makedirs(testPath + '/negatives/')
        for file in negList:
            shutil.copy(ndatPath + '/negatives/' + file, testPath + '/negatives/')

        # test the performance.
        print('---------------------------- Test Below ----------------------------')
        os.system("python patch_gnn.py")

    return

def Get_FileList(dataPath):
    # get negative list.
    negList = os.listdir(os.path.join(dataPath, 'negatives/0'))

    # get class label list.
    posClass = os.listdir(os.path.join(dataPath, 'positives'))
    numClass = len(posClass)

    # get positive list.
    posLists = {}
    for n in range(numClass):
        posLists[posClass[n]] = os.listdir(os.path.join(dataPath, 'positives/' + posClass[n]))

    return negList, posLists

if __name__ == '__main__':
    # initialize the log file.
    # logfile = 'analyze_subclass.txt'
    # if os.path.exists(os.path.join(logsPath, logfile)):
    #     os.remove(os.path.join(logsPath, logfile))
    # elif not os.path.exists(logsPath):
    #     os.makedirs(logsPath)
    # sys.stdout = Logger(os.path.join(logsPath, logfile))
    # main
    if (os.path.exists(ndatPath)):
        main()
    else:
        print(f'[WARNING] Cannot find the folder {ndatPath}')
