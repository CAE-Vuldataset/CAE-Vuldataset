'''
    Visualize the experimental records.
'''

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.set_printoptions(threshold=np.inf)

rootPath = './'
logsPath = rootPath + '/logs/'
logsFile = ''

_MAXXLIM = 100000
_SHOWFIG = False

def main():
    if os.path.exists(logsFile):
        logsFilePath = logsFile
    elif os.path.exists(os.path.join(logsPath, logsFile)):
        logsFilePath = os.path.join(logsPath, logsFile)
    else:
        print('[ERROR] Cannot find the file ', logsFile)
        return

    print('[INFO] Load the logs from ', logsFilePath)
    results = ParseData(logsFilePath)
    fig = DrawFigure(results)

    savefile = logsFilePath[:-4] + '.png'
    fig.savefig(savefile, dpi=fig.dpi)
    print('[INFO] Save the figure in ', savefile)

    return

def ParseData(logsFilePath):
    results = []
    # read file.
    f = open(logsFilePath)
    lines = f.readlines()
    # for each line.
    for line in lines:
        contents = re.findall(r'Epoch: \d+, Loss: \d+\.?\d*, Train Acc: \d+\.?\d*, Test Acc: \d+\.?\d*', line)
        if 0 == len(contents):
            continue
        # find 4 numbers.
        contents = re.findall(r'\d+\.?\d*', contents[0])
        cont2num = [float(content) for content in contents]
        results.append(cont2num)

    results = np.array(results)
    results = results.T
    return results

def DrawFigure(data):
    global _MAXXLIM
    if (len(data[0]) < _MAXXLIM):
        _MAXXLIM = len(data[0])
    fig = plt.figure(1)

    # ax1
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(data[0], data[1], color='blue', label='loss')
    plt.legend(loc='upper right')
    plt.xlim([0,_MAXXLIM])
    plt.grid(color='gray', linestyle=':')
    plt.ylabel('loss')


    # ax2
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(data[0], 100*data[2], color='red', label='train')
    ax2.plot(data[0], 100*data[3], color='green', label='test')
    plt.legend(loc='upper left')
    plt.xlim([0, _MAXXLIM])
    plt.grid(color='gray', linestyle=':')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')

    global _SHOWFIG
    if (_SHOWFIG):
        plt.show()
    return fig

def ArgsParser():
    # define argument parser.
    parser = argparse.ArgumentParser()
    # add arguments.
    parser.add_argument('-log', help='the log file name.', required=True)
    parser.add_argument('-maxepoch', help='the max epoch in x-axis.', type=int)
    parser.add_argument('-show', help='show the figure', action='store_true')
    # parse the arguments.
    args = parser.parse_args()
    # global variables.
    global logsFile
    global _MAXXLIM
    global _SHOWFIG
    # perform actions.
    if (args.log): logsFile = args.log
    if (args.maxepoch): _MAXXLIM = args.maxepoch
    if (args.show): _SHOWFIG = True

    return parser

if __name__ == '__main__':
    # sparse the arguments.
    ArgsParser()
    # main
    main()