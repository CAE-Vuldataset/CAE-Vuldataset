"""
    SecurityPatchIdentificationRNN: Security Patch Identification using RNN model.
    Developer: Shu Wang
    Date: 2020-08-08
    Version: S2020.08.08-V4
    Description: patch identification using both commit messages and normalized diff code.
    File Structure:
    SecurityPatchIdentificationRNN
        |-- analysis                                # task analysis.
        |-- data                                    # data storage.
                |-- negatives                           # negative samples.
                |-- positives                           # positive samples.
                |-- security_patch                      # positive samples. (official)
        |-- temp                                    # temporary stored variables.
                |-- data.npy                            # raw data. (important)
                |-- props.npy                           # properties of diff code. (important)
                |-- msgs.npy                            # commit messages. (important)
                |-- ...                                 # other temporary files. (trivial)
        |-- SecurityPatchIdentificationRNN.ipynb    # main entrance. (Google Colaboratory)
        |-- SecurityPatchIdentificationRNN.py       # main entrance. (Local)
    Usage:
        python SecurityPatchIdentificationRNN.py
    Dependencies:
        clang >= 6.0.0.2
        torch >= 1.2.0+cu92
        nltk  >= 3.3
"""

# dependencies.
import os

os.system("pip install clang")
import re
import gc
import math
import random
import numpy as np
import pandas as pd
import nltk
import time

# nltk.download("stopwords")
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import clang.cindex
import clang.enumerations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.metrics import accuracy_score

# environment settings.
_COLAB_ = (
    0 if (os.getenv("COLAB_GPU", "NONE") == "NONE") else 1
)  # 0 : Local environment, 1 : Google Colaboratory.
# file paths.
rootPath = "./drive/My Drive/Colab Notebooks/" if (_COLAB_) else "./"
dataPath = rootPath + "/zxh_data/"
# sDatPath = dataPath + "/security_patch/"
# pDatPath = dataPath + "/positives/"
# nDatPath = dataPath + "/negatives/"
trainPath = "/train/"
testPath = "/test/"
valPath = "/val/"
tempPath = rootPath + "/zxh_temp/"

# hyper-parameters. (affect GPU memory size)
_DiffEmbedDim_ = 128  # 128
_DiffMaxLen_ = 600  # 200(0.7), 314(0.8), 609(0.9), 1100(0.95), 2200(0.98), 3289(0.99), 5000(0.995), 10000(0.9997)
_DRnnHidSiz_ = 16  # 16
_MsgEmbedDim_ = 128  # 128
_MsgMaxLen_ = 200  # 54(0.9), 78(0.95), 130(0.98), 187(0.99), 268(0.995), 356(0.998), 516(0.999), 1434(1)
_MRnnHidSiz_ = 16  # 16
_TwinEmbedDim_ = 128  # 128
_TwinMaxLen_ = 800  # 224(0.8), 425(0.9), 755(0.95), 1448(0.98), 2270(0.99)
_TRnnHidSiz_ = 16  # 16
# hyper-parameters. (affect training speed)
_DRnnBatchSz_ = 128  # 128
_DRnnLearnRt_ = 0.0001  # 0.0001
_MRnnBatchSz_ = 128  # 128
_MRnnLearnRt_ = 0.0001  # 0.0001
_PRnnBatchSz_ = 256  # 256
_PRnnLearnRt_ = 0.0005  # 0.0005
_TRnnBatchSz_ = 256  # 256
_TRnnLearnRt_ = 0.0005  # 0.0005
# hyper-parameters. (trivial network parameters, unnecessary to modify)
_DiffExtraDim_ = 2  # 2
_TwinExtraDim_ = 1  # 1
_DRnnHidLay_ = 1  # 1
_MRnnHidLay_ = 1  # 1
_TRnnHidLay_ = 1  # 1
# hyper-parameters. (epoch related parameters, unnecessary to modify)
_DRnnMaxEpoch_ = 1000  # 1000
_DRnnPerEpoch_ = 1  # 1
_DRnnJudEpoch_ = 10  # 10
_MRnnMaxEpoch_ = 1000  # 1000
_MRnnPerEpoch_ = 1  # 1
_MRnnJudEpoch_ = 10  # 10
_PRnnMaxEpoch_ = 1000  # 1000
_PRnnPerEpoch_ = 1  # 1
_PRnnJudEpoch_ = 10  # 10
_TRnnMaxEpoch_ = 1000  # 1000
_TRnnPerEpoch_ = 1  # 1
_TRnnJudEpoch_ = 10  # 10
# hyper-parameters. (flow control)
_DEBUG_ = 0  #  0 : release
#  1 : debug
_LOCK_ = 0  #  0 : unlocked - create random split sets.
#  1 : locked   - use the saved split sets.
_MODEL_ = 1  #  0 : unlocked - train a new model.
#  1 : locked   - load the saved model.
_DTYP_ = 1  #  0 : maintain both diff code and context code.
#  1 : only maintain diff code.
_CTYP_ = 1  #  0 : maintain both the code and comments.
#  1 : only maintain code and delete comments.
_NIND_ = 1  # -1 : not abstract tokens. (and will disable _NLIT_)
#  0 : abstract identifiers with VAR/FUNC.
#  1 : abstract identifiers with VARn/FUNCn.
_NLIT_ = 1  #  0 : abstract literals with LITERAL.
#  1 : abstract literals with LITERAL/n.
_TWIN_ = 1  #  0 : only twin neural network.
#  1 : twins + msg neural network.

# print setting.
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)


def processDiff(diffProps, type):
    # only maintain the diff parts of the code.
    # diffProps = ProcessTokens(diffProps, dType=_DTYP_, cType=_CTYP_)
    # normalize the tokens of identifiers, literals, and comments.
    # diffProps = AbstractTokens(diffProps, iType=_NIND_, lType=_NLIT_)
    # get the diff token vocabulary.
    diffVocab, diffMaxLen = GetDiffVocab(diffProps, type)
    # get the max diff length.
    diffMaxLen = _DiffMaxLen_ if (diffMaxLen > _DiffMaxLen_) else diffMaxLen
    # get the diff token dictionary.
    diffDict = GetDiffDict(diffVocab)
    # get pre-trained weights for embedding layer.
    diffPreWeights = GetDiffEmbed(diffDict, _DiffEmbedDim_, type)
    # get the mapping for feature data and labels.
    diffData, diffLabels = GetDiffMapping(diffProps, diffMaxLen, diffDict, type)
    # change the tokentypes into one-hot vector.
    diffData = UpdateTokenTypes(diffData, type)
    return diffData, diffLabels



def merge_twinPreWeights(file_paths):


    files = []
    for root, _, filenames in os.walk(file_paths):
        for filename in filenames:
            if "twinPreWeights" in filename:
                files.append(os.path.join(root, filename))
    if not files:
        print("No files containing weights found.")
        return


    weights = [np.load(file_path) for file_path in files]


    merged_twinPreWeights = np.concatenate(weights, axis=0)

    print("[INFO] Weights merged and saved successfully!")
    return merged_twinPreWeights


def demoDiffRNN():
    """
    demo program of using diff code to identify patches.
    """

    # load data.
    train, test, val = ReadData()
    # if not os.path.exists(tempPath + "/data.npy"):  # | (not _DEBUG_)
    #     dataLoaded = ReadData()
    # else:
    #     dataLoaded = np.load(tempPath + "/data.npy", allow_pickle=True)
    #     print(
    #         "[INFO] <ReadData> Load "
    #         + str(len(dataLoaded))
    #         + " raw data from "
    #         + tempPath
    #         + "/data.npy."
    #     )

    # get the diff file properties.
    train_diffProps = GetDiffProps(train, "train")
    test_diffProps = GetDiffProps(test, "test")
    val_diffProps = GetDiffProps(val, "val")
    # if not os.path.exists(tempPath + "/props.npy"):
    #     diffProps = GetDiffProps(dataLoaded)
    # else:
    #     diffProps = np.load(tempPath + "/props.npy", allow_pickle=True)
    #     print(
    #         "[INFO] <GetDiffProps> Load "
    #         + str(len(diffProps))
    #         + " diff property data from "
    #         + tempPath
    #         + "/props.npy."
    #     )

    # # only maintain the diff parts of the code.
    # # diffProps = ProcessTokens(diffProps, dType=_DTYP_, cType=_CTYP_)
    # # normalize the tokens of identifiers, literals, and comments.
    # # diffProps = AbstractTokens(diffProps, iType=_NIND_, lType=_NLIT_)
    # # get the diff token vocabulary.
    # diffVocab, diffMaxLen = GetDiffVocab(diffProps)
    # # get the max diff length.
    # diffMaxLen = _DiffMaxLen_ if (diffMaxLen > _DiffMaxLen_) else diffMaxLen
    # # get the diff token dictionary.
    # diffDict = GetDiffDict(diffVocab)
    # # get pre-trained weights for embedding layer.
    # diffPreWeights = GetDiffEmbed(diffDict, _DiffEmbedDim_)
    # # get the mapping for feature data and labels.
    # diffData, diffLabels = GetDiffMapping(diffProps, diffMaxLen, diffDict)
    # # change the tokentypes into one-hot vector.
    # diffData = UpdateTokenTypes(diffData)

    dataTrain, labelTrain = processDiff(train_diffProps, "train")
    dataTest, labelTest = processDiff(test_diffProps, "test")
    dataValid, labelValid = processDiff(val_diffProps, "val")

    diffPreWeights = merge_twinPreWeights(tempPath)

    # split data into rest/test dataset.
    # dataRest, labelRest, dataTest, labelTest = SplitData(
    #     diffData, diffLabels, "test", rate=0.2
    # )
    # # split data into train/valid dataset.
    # dataTrain, labelTrain, dataValid, labelValid = SplitData(
    #     dataRest, labelRest, "valid", rate=0.2
    # )
    # print(
    #     "[INFO] <main> Get "
    #     + str(len(dataTrain))
    #     + " TRAIN data, "
    #     + str(len(dataValid))
    #     + " VALID data, "
    #     + str(len(dataTest))
    #     + " TEST data. (Total: "
    #     + str(len(dataTrain) + len(dataValid) + len(dataTest))
    #     + ")"
    # )

    # DiffRNNTrain
    # if (_MODEL_) & (os.path.exists(tempPath + "/model_DiffRNN.pth")):
    #     preWeights = torch.from_numpy(diffPreWeights)
    #     model = DiffRNN(preWeights, hiddenSize=_DRnnHidSiz_, hiddenLayers=_DRnnHidLay_)
    #     model.load_state_dict(torch.load(tempPath + "/model_DiffRNN.pth"))
    # else:
    model = DiffRNNTrain(
        dataTrain,
        labelTrain,
        dataValid,
        labelValid,
        preWeights=diffPreWeights,
        batchsize=_DRnnBatchSz_,
        learnRate=_DRnnLearnRt_,
        dTest=dataTest,
        lTest=labelTest,
    )

    # DiffRNNTest
    predictions, accuracy = DiffRNNTest(
        model, dataTest, labelTest, batchsize=_DRnnBatchSz_
    )
    _, confusion = OutputEval(predictions, labelTest, "DiffRNN")

    print(dataset)
    return


def ReadData():
    """
    Read data from the files.
    :return: data - a set of commit message, diff code, and labels.
    [[['', ...], [['', ...], ['', ...], ...], 0/1], ...]
    """

    def ReadCommitMsg(filename):
        """
        Read commit message from a file.
        :param filename: file name (string).
        :return: commitMsg - commit message.
        ['line', 'line', ...]
        """

        fp = open(filename, encoding="utf-8", errors="ignore")  # get file point.
        lines = fp.readlines()  # read all lines.
        # numLines = len(lines)   # get the line number.
        # print(lines)

        # initialize commit message.
        commitMsg = []
        # get the wide range of commit message.
        for line in lines:
            if line.startswith("diff --git"):
                break
            else:
                commitMsg.append(line)
        # print(commitMsg)
        # process the head of commit message.
        while 1:
            headMsg = commitMsg[0]
            if (
                headMsg.startswith("From")
                or headMsg.startswith("Date:")
                or headMsg.startswith("Subject:")
                or headMsg.startswith("commit")
                or headMsg.startswith("Author:")
            ):
                commitMsg.pop(0)
            else:
                break
        # print(commitMsg)
        # process the tail of commit message.
        dashLines = [
            i for i in range(len(commitMsg)) if commitMsg[i].startswith("---")
        ]  # finds all lines start with ---.
        if len(dashLines):
            lnum = dashLines[-1]  # last line number of ---
            marks = [
                (
                    1
                    if (
                        " file changed, " in commitMsg[i]
                        or " files changed, " in commitMsg[i]
                    )
                    else 0
                )
                for i in range(lnum, len(commitMsg))
            ]
            if sum(marks):
                for i in reversed(range(lnum, len(commitMsg))):
                    commitMsg.pop(i)
        # print(commitMsg)

        # msgShow = ''
        # for i in range(len(commitMsg)):
        #    msgShow += commitMsg[i]
        # print(msgShow)

        return commitMsg

    def ReadDiffLines(filename):
        """
        Read diff code from a file.
        :param filename:  file name (string).
        :return: diffLines - diff code.
        [['line', ...], ['line', ...], ...]
        """

        fp = open(filename, encoding="utf-8", errors="ignore")  # get file point.
        lines = fp.readlines()  # read all lines.
        numLines = len(lines)  # get the line number.
        # print(lines)

        atLines = [
            i for i in range(numLines) if lines[i].startswith("@@ ")
        ]  # find all lines start with @@.
        atLines.append(numLines)
        # print(atLines)

        diffLines = []
        for nh in range(len(atLines) - 1):  # find all hunks.
            # print(atLines[nh], atLines[nh + 1])
            hunk = []
            for nl in range(atLines[nh] + 1, atLines[nh + 1]):
                # print(lines[nl], end='')
                if lines[nl].startswith("diff --git "):
                    break
                else:
                    hunk.append(lines[nl])
            diffLines.append(hunk)
            # print(hunk)
        # print(diffLines)
        # print(filename)
        # print(diffLines)
        # print(len(diffLines))

        # process the last hunk.
        # print(filename)
        lastHunk = diffLines[-1]
        numLastHunk = len(lastHunk)
        dashLines = [i for i in range(numLastHunk) if lastHunk[i].startswith("--")]
        if len(dashLines):
            lnum = dashLines[-1]
            for i in reversed(range(lnum, numLastHunk)):
                lastHunk.pop(i)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(diffLines)
        # print(len(diffLines))

        return diffLines

    # create temp folder.
    if not os.path.exists(tempPath):
        os.makedirs(tempPath)
    fp = open(tempPath + "/filelist.txt", "w")

    # initialize data.
    # data = []
    train = []
    test = []
    val = []
    # read train data.
    print(dataPath + dataset + trainPath)
    for root, ds, fs in os.walk(dataPath + dataset + trainPath):

        for file in fs:
            filename = os.path.join(root, file).replace("\\", "/")
            fp.write(filename + "\n")
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            train.append(
                [commitMsg, diffLines, int(os.path.basename(file).split(".")[0])]
            )

    # read test data.
    for root, ds, fs in os.walk(dataPath + test_val + testPath):
        for file in fs:
            filename = os.path.join(root, file).replace("\\", "/")
            fp.write(filename + "\n")
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            test.append(
                [commitMsg, diffLines, int(os.path.basename(file).split(".")[0])]
            )

    # read val data.
    for root, ds, fs in os.walk(dataPath + test_val + valPath):
        for file in fs:
            filename = os.path.join(root, file).replace("\\", "/")
            fp.write(filename + "\n")
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            val.append(
                [commitMsg, diffLines, int(os.path.basename(file).split(".")[0])]
            )
    fp.close()

    # print(len(dataLoaded))
    # print(len(dataLoaded[0]))
    # print(dataLoaded)
    # [[['a', 'b', 'c', ], [['', '', '', ], ['', '', '', ], ], 0/1], ]
    # sample = dataLoaded[i]
    # commitMsg = dataLoaded[i][0]
    # diffLines = dataLoaded[i][1]
    # label = dataLoaded[i][2]
    # diffHunk = dataLoaded[i][1][j]

    # save dataLoaded.
    # if not os.path.exists(tempPath + "/train.npy"):
    #     np.save(tempPath + "/train.npy", train, allow_pickle=True)
    #     print(
    #         "[INFO] <ReadData> Save "
    #         + str(len(train))
    #         + " raw data to "
    #         + tempPath
    #         + "/train.npy."
    #     )
    # if not os.path.exists(tempPath + "/test.npy"):
    #     np.save(tempPath + "/test.npy", test, allow_pickle=True)
    #     print(
    #         "[INFO] <ReadData> Save "
    #         + str(len(test))
    #         + " raw data to "
    #         + tempPath
    #         + "/test.npy."
    #     )
    # if not os.path.exists(tempPath + "/val.npy"):
    #     np.save(tempPath + "/val.npy", val, allow_pickle=True)
    #     print(
    #         "[INFO] <ReadData> Save "
    #         + str(len(val))
    #         + " raw data to "
    #         + tempPath
    #         + "/val.npy."
    #     )

    return train, test, val
    # return test


def GetDiffProps(data, type):
    """
    Get the properties of the code in diff files.
    :param data: [[[line, , ], [[line, , ], [line, , ], ...], 0/1], ...]
    :return: props - [[[tokens], [nums], [nums], 0/1], ...]
    """

    def RemoveSign(line):
        """
        Remove the sign (+/-) in the first character.
        :param line: a code line.
        :return: process line.
        """

        return " " + line[1:] if (line[0] == "+") or (line[0] == "-") else line

    def GetClangTokens(line):
        """
        Get the tokens of a line with the Clang tool.
        :param line: a code line.
        :return: tokens - ['tk', 'tk', ...] ('tk': string)
                 tokenTypes - [tkt, tkt, ...] (tkt: 1, 2, 3, 4, 5)
                 diffTypes - [dft, dft, ...] (dft: -1, 0, 1)
        """

        # remove non-ascii
        line = line.encode("ascii", "ignore").decode()

        # defination.
        tokenClass = [
            clang.cindex.TokenKind.KEYWORD,  # 1
            clang.cindex.TokenKind.IDENTIFIER,  # 2
            clang.cindex.TokenKind.LITERAL,  # 3
            clang.cindex.TokenKind.PUNCTUATION,  # 4
            clang.cindex.TokenKind.COMMENT,
        ]  # 5
        tokenDict = {cls: index + 1 for index, cls in enumerate(tokenClass)}
        # print(tokenDict)

        # initialize.
        tokens = []
        tokenTypes = []
        diffTypes = []

        # clang sparser.
        idx = clang.cindex.Index.create()
        tu = idx.parse(
            "tmp.cpp",
            args=["-std=c++11"],
            unsaved_files=[("tmp.cpp", RemoveSign(line))],
            options=0,
        )
        for t in tu.get_tokens(extent=tu.cursor.extent):
            # print(t.kind, t.spelling, t.location)
            tokens.append(t.spelling)
            tokenTypes.append(tokenDict[t.kind])
            diffTypes.append(1 if (line[0] == "+") else -1 if (line[0] == "-") else 0)
        # print(tokens)
        # print(tokenTypes)
        # print(diffTypes)

        return tokens, tokenTypes, diffTypes

    def GetWordTokens(line):
        """
        Get the word tokens from a code line.
        :param line: a code line.
        :return: tokens - ['tk', 'tk', ...] ('tk': string)
        """

        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(RemoveSign(line))
        return tokens

    def GetString(lines):
        """
        Get the strings from the diff code
        :param lines: diff code.
        :return: lineStr - All the diff lines.
                 lineStrB - The before-version code lines.
                 lineStrA - The after-version code lines.
        """

        lineStr = ""
        lineStrB = ""
        lineStrA = ""
        for hunk in lines:
            for line in hunk:
                # all lines.
                lineStr += RemoveSign(line)
                # all Before lines.
                lineStrB += RemoveSign(line) if line[0] != "+" else ""
                # all After lines.
                lineStrA += RemoveSign(line) if line[0] != "-" else ""

        return lineStr, lineStrB, lineStrA

    def GetDiffTokens(lines):
        """
        Get the tokens for the diff lines.
        :param lines: the diff code.
        :return: tokens - tokens ['tk', 'tk', ...] ('tk': string)
                 tokenTypes - token types [tkt, tkt, ...] (tkt: 1, 2, 3, 4, 5)
                 diffTypes - diff types [dft, dft, ...] (dft: -1, 0, 1)
        """

        # initialize.
        tokens = []
        tokenTypes = []
        diffTypes = []

        # for each line of lines.
        for hunk in lines:
            for line in hunk:
                # print(line, end='')
                tk, tkT, dfT = GetClangTokens(line)
                tokens.extend(tk)
                tokenTypes.extend(tkT)
                diffTypes.extend(dfT)
                # print('-----------------------------------------------------------------------')
        # print(tokens)
        # print(tokenTypes)
        # print(diffTypes)

        return tokens, tokenTypes, diffTypes

    # lines = data[0][1]
    # print(lines)
    # hunk = data[0][1][0]
    # print(hunk)
    # line = data[0][1][0][0]
    # print(line)

    # for each sample data[n].
    numData = len(data)
    props = []
    for n in range(numData):
        # get the lines of the diff file.
        diffLines = data[n][1]
        # properties.
        tk, tkT, dfT = GetDiffTokens(diffLines)
        label = data[n][2]
        prop = [tk, tkT, dfT, label]
        # print(prop)
        props.append(prop)
        print(n)

    # save props.
    # if not os.path.exists(tempPath):
    #     os.mkdir(tempPath)
    # if not os.path.exists(tempPath + f"/{type}_props.npy"):
    #     np.save(tempPath + f"/{type}_props.npy", props, allow_pickle=True)
    #     print(
    #         "[INFO] <GetDiffProps> Save "
    #         + str(len(props))
    #         + " diff property data to "
    #         + tempPath
    #         + f"/{type}_props.npy."
    #     )

    return props


def GetDiffVocab(props, type):
    """
    Get the vocabulary of diff tokens
    :param props - the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :return: vocab - the vocabulary of diff tokens. ['tk', 'tk', ...]
             maxLen - the max length of a diff code.
    """

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    fp = open(tempPath + f"/{type}_difflen.csv", "w")

    # get the whole tokens and the max diff length.
    tokens = []
    maxLen = 0

    # for each sample.
    for item in props:
        tokens.extend(item[0])
        maxLen = len(item[0]) if (len(item[0]) > maxLen) else maxLen
        fp.write(str(len(item[0])) + "\n")
    fp.close()

    # remove duplicates and get vocabulary.
    vocab = {}.fromkeys(tokens)
    vocab = list(vocab.keys())

    # print.
    print(
        "[INFO] <GetDiffVocab> There are "
        + str(len(vocab))
        + " diff vocabulary tokens. (except '<pad>')"
    )
    print(
        "[INFO] <GetDiffVocab> The max diff length is "
        + str(maxLen)
        + " tokens. (hyperparameter: _DiffMaxLen_ = "
        + str(_DiffMaxLen_)
        + ")"
    )

    return vocab, maxLen


def GetDiffDict(vocab):
    """
    Get the dictionary of diff vocabulary.
    :param vocab: the vocabulary of diff tokens. ['tk', 'tk', ...]
    :return: tokenDict - the dictionary of diff vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    """

    # get token dict from vocabulary.
    tokenDict = {token: (index + 1) for index, token in enumerate(vocab)}
    tokenDict["<pad>"] = 0

    # print.
    print(
        "[INFO] <GetDiffDict> Create dictionary for "
        + str(len(tokenDict))
        + " diff vocabulary tokens. (with '<pad>')"
    )

    return tokenDict


def GetDiffEmbed(tokenDict, embedSize, type):
    """
    Get the pre-trained weights for embedding layer from the dictionary of diff vocabulary.
    :param tokenDict: the dictionary of diff vocabulary.
    {'tk': 0, 'tk': 1, ..., '<pad>': N}
    :param embedSize: the dimension of the embedding vector.
    :return: preWeights - the pre-trained weights for embedding layer.
    [[n, ...], [n, ...], ...]
    """

    # number of the vocabulary tokens.
    numTokens = len(tokenDict)

    # initialize the pre-trained weights for embedding layer.
    preWeights = np.zeros((numTokens, embedSize))
    for index in range(numTokens):
        preWeights[index] = np.random.normal(size=(embedSize,))
    print(
        "[INFO] <GetDiffEmbed> Create pre-trained embedding weights with "
        + str(len(preWeights))
        + " * "
        + str(len(preWeights[0]))
        + " matrix."
    )

    # save preWeights.
    if not os.path.exists(tempPath + f"/{type}_twinPreWeights.npy"):
        np.save(tempPath + f"/{type}_twinPreWeights.npy", preWeights, allow_pickle=True)
        print(
            "[INFO] <GetDiffEmbed> Save the pre-trained weights of embedding layer to "
            + tempPath
            + f"/{type}_twinPreWeights.npy"
        )

    return preWeights


def GetDiffMapping(props, maxLen, tokenDict, type):
    """
    Map the feature data into indexed data.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :param maxLen: the max length of a diff code.
    :param tokenDict: the dictionary of diff vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    :return: np.array(data) - feature data.
             [[[n, {0~5}, {-1~1}], ...], ...]
             np.array(labels) - labels.
             [[0/1], ...]
    """

    def PadList(dList, pad, length):
        """
        Pad the list data to a fixed length.
        :param dList: the list data - [ , , ...]
        :param pad: the variable used to pad.
        :param length: the fixed length.
        :return: dList - padded list data. [ , , ...]
        """

        if len(dList) <= length:
            dList.extend(pad for i in range(length - len(dList)))
        elif len(dList) > length:
            dList = dList[0:length]

        return dList

    # initialize the data and labels.
    data = []
    labels = []

    # for each sample.
    for item in props:
        # initialize sample.
        sample = []

        # process token.
        tokens = item[0]
        tokens = PadList(tokens, "<pad>", maxLen)
        tokens2index = []
        for tk in tokens:
            tokens2index.append(tokenDict[tk])
        sample.append(tokens2index)
        # process tokenTypes.
        tokenTypes = item[1]
        tokenTypes = PadList(tokenTypes, 0, maxLen)
        sample.append(tokenTypes)
        # process diffTypes.
        diffTypes = item[2]
        diffTypes = PadList(diffTypes, 0, maxLen)
        sample.append(diffTypes)

        # process sample.
        sample = np.array(sample).T
        data.append(sample)
        # process label.
        label = item[3]
        labels.append([label])

    if _DEBUG_:
        print("[DEBUG] data:")
        print(data[0:3])
        print("[DEBUG] labels:")
        print(labels[0:3])

    # print.
    print(
        "[INFO] <GetDiffMapping> Create "
        + str(len(data))
        + " feature data with "
        + str(len(data[0]))
        + " * "
        + str(len(data[0][0]))
        + " matrix."
    )
    print(
        "[INFO] <GetDiffMapping> Create "
        + str(len(labels))
        + " labels with 1 * 1 matrix."
    )

    # save files.
    if (not os.path.exists(tempPath + f"/{type}_ndata_" + str(maxLen) + ".npy")) | (
        not os.path.exists(tempPath + f"/{type}_nlabels_" + str(maxLen) + ".npy")
    ):
        np.save(
            tempPath + f"/{type}_ndata_" + str(maxLen) + ".npy", data, allow_pickle=True
        )
        print(
            "[INFO] <GetDiffMapping> Save the mapped numpy data to "
            + tempPath
            + f"/{type}_ndata_"
            + str(maxLen)
            + ".npy."
        )
        np.save(
            tempPath + f"/{type}_nlabels_" + str(maxLen) + ".npy",
            labels,
            allow_pickle=True,
        )
        print(
            "[INFO] <GetDiffMapping> Save the mapped numpy labels to "
            + tempPath
            + f"/{type}_nlabels_"
            + str(maxLen)
            + ".npy."
        )

    return np.array(data), np.array(labels)


def UpdateTokenTypes(data, type):
    """
    Update the token type in the feature data into one-hot vector.
    :param data: feature data. [[[n, {0~5}, {-1~1}], ...], ...]
    :return: np.array(newData). [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    """

    newData = []
    # for each sample.
    for item in data:
        # get the transpose of props.
        itemT = item.T
        # initialize new sample.
        newItem = []
        newItem.append(itemT[0])
        newItem.extend(np.zeros((5, len(item)), dtype=int))
        newItem.append(itemT[2])
        # assign the new sample.
        for i in range(len(item)):
            tokenType = itemT[1][i]
            if tokenType:
                newItem[tokenType][i] = 1
        # get the transpose of new sample.
        newItem = np.array(newItem).T
        # append new sample.
        newData.append(newItem)

    if _DEBUG_:
        print("[DEBUG] newData:")
        print(newData[0:3])

    # print.
    print(
        "[INFO] <UpdateTokenTypes> Update "
        + str(len(newData))
        + " feature data with "
        + str(len(newData[0]))
        + " * "
        + str(len(newData[0][0]))
        + " matrix."
    )

    # save files.
    if not os.path.exists(
        tempPath + f"/{type}_newdata_" + str(len(newData[0])) + ".npy"
    ):
        np.save(
            tempPath + f"/{type}_newdata_" + str(len(newData[0])) + ".npy",
            newData,
            allow_pickle=True,
        )
        print(
            "[INFO] <UpdateTokenTypes> Save the mapped numpy data to "
            + tempPath
            + f"/{type}_newdata_"
            + str(len(newData[0]))
            + ".npy."
        )

    # change marco.
    global _DiffExtraDim_
    _DiffExtraDim_ = 6

    return np.array(newData)


def SplitData(data, labels, setType, rate=0.2):
    """
    Split the data and labels into two sets with a specific rate.
    :param data: feature data.
    [[[n, {0~5}, {-1~1}], ...], ...]
    [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param labels: labels. [[0/1], ...]
    :param setType: the splited dataset type.
    :param rate: the split rate. 0 ~ 1
    :return: dsetRest - the rest dataset.
             lsetRest - the rest labels.
             dset - the splited dataset.
             lset - the splited labels.
    """

    # set parameters.
    setType = setType.upper()
    numData = len(data)
    num = math.floor(numData * rate)

    # get the random data list.
    if (os.path.exists(tempPath + "/split_" + setType + ".npy")) & (_LOCK_):
        dataList = np.load(tempPath + "/split_" + setType + ".npy")
    else:
        dataList = list(range(numData))
        random.shuffle(dataList)
        np.save(tempPath + "/split_" + setType + ".npy", dataList, allow_pickle=True)

    # split data.
    dset = data[dataList[0:num]]
    lset = labels[dataList[0:num]]
    dsetRest = data[dataList[num:]]
    lsetRest = labels[dataList[num:]]

    # print.
    setTypeRest = "TRAIN" if (setType == "VALID") else "REST"
    print(
        "[INFO] <SplitData> Split data into "
        + str(len(dsetRest))
        + " "
        + setTypeRest
        + " dataset and "
        + str(len(dset))
        + " "
        + setType
        + " dataset. (Total: "
        + str(len(dsetRest) + len(dset))
        + ", Rate: "
        + str(int(rate * 100))
        + "%)"
    )

    return dsetRest, lsetRest, dset, lset


class DiffRNN(nn.Module):
    """
    DiffRNN : convert a text data into a predicted label.
    """

    def __init__(self, preWeights, hiddenSize=32, hiddenLayers=1):
        """
        define each layer in the network model.
        :param preWeights: tensor pre-trained weights for embedding layer.
        :param hiddenSize: node number in the hidden layer.
        :param hiddenLayers: number of hidden layer.
        """

        super(DiffRNN, self).__init__()
        # parameters.
        class_num = 2
        vocabSize, embedDim = preWeights.size()
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocabSize, embedding_dim=embedDim)
        self.embedding.load_state_dict({"weight": preWeights})
        self.embedding.weight.requires_grad = True
        # LSTM Layer
        # _DiffExtraDim_ = 6
        if _DEBUG_:
            print(_DiffExtraDim_)
        self.lstm = nn.LSTM(
            input_size=embedDim + _DiffExtraDim_,
            hidden_size=hiddenSize,
            num_layers=hiddenLayers,
            bidirectional=True,
        )
        # Fully-Connected Layer
        self.fc = nn.Linear(hiddenSize * hiddenLayers * 2, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * diff_length * feature_dim.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        """

        # x             batch_size * diff_length * feature_dim
        embeds = self.embedding(x[:, :, 0])
        # embeds        batch_size * diff_length * embedding_dim
        features = x[:, :, 1:]
        # features      batch_size * diff_length * _DiffExtraDim_
        inputs = torch.cat((embeds.float(), features.float()), 2)
        # inputs        batch_size * diff_length * (embedding_dim + _DiffExtraDim_)
        inputs = inputs.permute(1, 0, 2)
        # inputs        diff_length * batch_size * (embedding_dim + _DiffExtraDim_)
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        # lstm_out      diff_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # feature_map   batch_size * (hidden_size * num_layers * direction_num)
        final_out = self.fc(feature_map)  # batch_size * class_num
        return self.softmax(final_out)  # batch_size * class_num


def DiffRNNTrain(
    dTrain,
    lTrain,
    dValid,
    lValid,
    preWeights,
    batchsize=64,
    learnRate=0.001,
    dTest=None,
    lTest=None,
):
    """
    Train the DiffRNN model.
    :param dTrain: training data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param lTrain: training label. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param dValid: validation data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param lValid: validation label. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param preWeights: pre-trained weights for embedding layer.
    :param batchsize: number of samples in a batch.
    :param learnRate: learning rate.
    :param dTest: test data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param lTest: test label. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :return: model - the DiffRNN model.
    """

    # get the mark of the test dataset.
    if dTest is None:
        dTest = []
    if lTest is None:
        lTest = []
    markTest = 1 if (len(dTest)) & (len(lTest)) else 0

    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    if markTest:
        xTest = torch.from_numpy(dTest).long().cuda()
        yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)
    if markTest:
        test = torchdata.TensorDataset(xTest, yTest)
        testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # get training weights.
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(2):
        weights.append(1 - lbTrain.count(lb) / len(lbTrain))
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWeights = torch.from_numpy(preWeights)
    model = DiffRNN(preWeights, hiddenSize=_DRnnHidSiz_, hiddenLayers=_DRnnHidLay_)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(
        "[INFO] <DiffRNNTrain> ModelType: DiffRNN, HiddenNodes: %d, HiddenLayers: %d."
        % (_DRnnHidSiz_, _DRnnHidLay_)
    )
    print(
        "[INFO] <DiffRNNTrain> BatchSize: %d, LearningRate: %.4f, MaxEpoch: %d, PerEpoch: %d."
        % (batchsize, learnRate, _DRnnMaxEpoch_, _DRnnPerEpoch_)
    )
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(_DRnnMaxEpoch_):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # back propagation.
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                # data conversion.
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                # forward propagation.
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # testing phase.
        if markTest:
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for iter, (data, label) in enumerate(testloader):
                    # data conversion.
                    data = data.to(device)
                    label = label.contiguous().view(-1)
                    label = label.to(device)
                    # forward propagation.
                    yhat = model.forward(data)  # get output
                    # statistic
                    preds = yhat.max(1)[1]
                    predictions.extend(preds.int().tolist())
                    labels.extend(label.int().tolist())
                    torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            # test accuracy.
            accTest = accuracy_score(labels, predictions) * 100

        # output information.
        if 0 == (epoch + 1) % _DRnnPerEpoch_:
            strAcc = "[Epoch {:03}] loss: {:.3}, train acc: {:.3f}%, valid acc: {:.3f}%.".format(
                epoch + 1, lossTrain, accTrain, accValid
            )
            if markTest:
                strAcc = strAcc[:-1] + ", test acc: {:.3f}%.".format(accTest)
            print(strAcc)
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + "/model_DiffRNN.pth")
        # stop judgement.
        if (epoch >= _DRnnJudEpoch_) and (
            accList[-1] < min(accList[-1 - _DRnnJudEpoch_ : -1])
        ):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + "/model_DiffRNN.pth"))
    print(
        "[INFO] <DiffRNNTrain> Finish training DiffRNN model. (Best model: "
        + tempPath
        + "/model_DiffRNN.pth)"
    )

    return model


def DiffRNNTest(model, dTest, lTest, batchsize=64):
    """
    Test the DiffRNN model.
    :param model: deep learning model.
    :param dTest: test data.
    :param lTest: test label.
    :param batchsize: number of samples in a batch
    :return: predictions - predicted labels. [[0], [1], ...]
             accuracy - the total test accuracy. numeric
    """

    # tensor data processing.
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    return predictions, accuracy


def OutputEval(predictions, labels, method=""):
    """
    Output the evaluation results.
    :param predictions: predicted labels. [[0], [1], ...]
    :param labels: ground truth labels. [[1], [1], ...]
    :param method: method name. string
    :return: accuracy - the total accuracy. numeric
             confusion - confusion matrix [[1000, 23], [12, 500]]
    """

    # evaluate the predictions with gold labels, and get accuracy and confusion matrix.
    def Evaluation(predictions, labels):

        # parameter settings.
        D = len(labels)
        cls = 2

        # get confusion matrix.
        confusion = np.zeros((cls, cls))
        for ind in range(D):
            nRow = int(predictions[ind][0])
            nCol = int(labels[ind][0])
            confusion[nRow][nCol] += 1

        # get accuracy.
        accuracy = 0
        for ind in range(cls):
            accuracy += confusion[ind][ind]
        accuracy /= D

        return accuracy, confusion

    # get accuracy and confusion matrix.
    accuracy, confusion = Evaluation(predictions, labels)
    precision = (
        confusion[1][1] / (confusion[1][0] + confusion[1][1])
        if (confusion[1][0] + confusion[1][1])
        else 0
    )
    recall = (
        confusion[1][1] / (confusion[0][1] + confusion[1][1])
        if (confusion[0][1] + confusion[1][1])
        else 0
    )
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # output on screen and to file.
    print("       -------------------------------------------")
    print("       method           :  " + method) if len(method) else print("", end="")
    print("       accuracy  (ACC)  :  %.3f%%" % (accuracy * 100))
    print("       precision (P)    :  %.3f%%" % (precision * 100))
    print("       recall    (R)    :  %.3f%%" % (recall * 100))
    print("       F1 score  (F1)   :  %.3f" % (F1))
    print(
        "       fall-out  (FPR)  :  %.3f%%"
        % (confusion[1][0] * 100 / (confusion[1][0] + confusion[0][0]))
    )
    print(
        "       miss rate (FNR)  :  %.3f%%"
        % (confusion[0][1] * 100 / (confusion[0][1] + confusion[1][1]))
    )
    print("       confusion matrix :      (actual)")
    print("                           Neg         Pos")
    print(
        "       (predicted) Neg     %-5d(TN)   %-5d(FN)"
        % (confusion[0][0], confusion[0][1])
    )
    print(
        "                   Pos     %-5d(FP)   %-5d(TP)"
        % (confusion[1][0], confusion[1][1])
    )
    TN, FN = confusion[0][0], confusion[0][1]
    FP, TP = confusion[1][0], confusion[1][1]
    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    print("       MCC  (MCC)   :  %.3f" % (MCC))
    print("       -------------------------------------------")

    return accuracy, confusion


def demoCommitMsg():
    """
    demo program of using commit message to identify patches.
    """

    # load data.
    if not os.path.exists(tempPath + "/data.npy"):  # | (not _DEBUG_)
        dataLoaded = ReadData()
    else:
        dataLoaded = np.load(tempPath + "/data.npy", allow_pickle=True)
        print(
            "[INFO] <ReadData> Load "
            + str(len(dataLoaded))
            + " raw data from "
            + tempPath
            + "/data.npy."
        )

    # get the commit messages from data.
    if not os.path.exists(tempPath + "/msgs.npy"):
        commitMsgs = GetCommitMsgs(dataLoaded)
    else:
        commitMsgs = np.load(tempPath + "/msgs.npy", allow_pickle=True)
        print(
            "[INFO] <GetCommitMsg> Load "
            + str(len(commitMsgs))
            + " commit messages from "
            + tempPath
            + "/msgs.npy."
        )

    # get the message token vocabulary.
    msgVocab, msgMaxLen = GetMsgVocab(commitMsgs)
    # get the max msg length.
    msgMaxLen = _MsgMaxLen_ if (msgMaxLen > _MsgMaxLen_) else msgMaxLen
    # get the msg token dictionary.
    msgDict = GetMsgDict(msgVocab)
    # get pre-trained weights for embedding layer.
    msgPreWeights = GetMsgEmbed(msgDict, _MsgEmbedDim_)
    # get the mapping for feature data and labels.
    msgData, msgLabels = GetMsgMapping(commitMsgs, msgMaxLen, msgDict)
    # split data into rest/test dataset.
    mdataTrain, mlabelTrain, mdataTest, mlabelTest = SplitData(
        msgData, msgLabels, "test", rate=0.2
    )

    # MsgRNNTrain
    if (_MODEL_) & (os.path.exists(tempPath + "/model_MsgRNN.pth")):
        preWeights = torch.from_numpy(msgPreWeights)
        model = MsgRNN(preWeights, hiddenSize=_MRnnHidSiz_, hiddenLayers=_MRnnHidLay_)
        model.load_state_dict(torch.load(tempPath + "/model_MsgRNN.pth"))
    else:
        model = MsgRNNTrain(
            mdataTrain,
            mlabelTrain,
            mdataTest,
            mlabelTest,
            msgPreWeights,
            batchsize=_MRnnBatchSz_,
            learnRate=_MRnnLearnRt_,
            dTest=mdataTest,
            lTest=mlabelTest,
        )

    # MsgRNNTest
    predictions, accuracy = MsgRNNTest(
        model, mdataTest, mlabelTest, batchsize=_MRnnBatchSz_
    )
    _, confusion = OutputEval(predictions, mlabelTest, "MsgRNN")

    return


def GetCommitMsgs(data, type):
    """
    Get the commit messages in diff files.
    :param data: [[[line, , ], [[line, , ], [line, , ], ...], 0/1], ...]
    :return: msgs - [[[tokens], 0/1], ...]
    """

    def GetMsgTokens(lines):
        """
        Get the tokens from a commit message.
        :param lines: commit message. [line, , ]
        :return: tokensStem ['tk', , ]
        """

        # concatenate lines.
        # get the string of commit message.
        msg = ""
        for line in lines:
            msg += line[:-1] + " "
        # print(msg)

        # pre-process.
        # remove url.
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        msg = re.sub(pattern, " ", msg)
        # remove independent numbers.
        pattern = r" \d+ "
        msg = re.sub(pattern, " ", msg)
        # lower case capitalized words.
        pattern = r"([A-Z][a-z]+)"

        def LowerFunc(matched):
            return matched.group(1).lower()

        msg = re.sub(pattern, LowerFunc, msg)
        # remove footnote.
        patterns = [
            "signed-off-by:",
            "reported-by:",
            "reviewed-by:",
            "acked-by:",
            "found-by:",
            "tested-by:",
            "cc:",
        ]
        for pattern in patterns:
            index = msg.find(pattern)
            if index > 0:
                msg = msg[:index]
        # print(msg)

        # clearance.
        # get the tokens.
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(msg)
        # clear tokens that don't contain any english letter.
        for i in reversed(range(len(tokens))):
            if not (re.search("[a-z]", tokens[i])):
                tokens.pop(i)
        # clear tokens that are stopwords.
        for i in reversed(range(len(tokens))):
            if tokens[i] in stopwords.words("english"):
                tokens.pop(i)
        pattern = re.compile("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
        for i in reversed(range(len(tokens))):
            if pattern.findall(tokens[i]):
                tokens.pop(i)
        # print(tokens)

        # process tokens with stemming.
        porter = PorterStemmer()
        tokensStem = []
        for item in tokens:
            tokensStem.append(porter.stem(item))
        # print(tokensStem)

        return tokensStem

    # for each sample data[n].
    numData = len(data)
    msgs = []
    for n in range(numData):
        # get the lines of the commit message.
        commitMsg = data[n][0]
        mtk = GetMsgTokens(commitMsg)
        # get the label.
        label = data[n][2]
        # print([mtk, label])
        # append the message tokens.
        msgs.append([mtk, label])
        print(n)

    # save commit messages.
    # if not os.path.exists(tempPath):
    #     os.mkdir(tempPath)
    # if not os.path.exists(tempPath + f"/{type}_msgs.npy"):
    #     np.save(tempPath + f"/{type}_msgs.npy", msgs, allow_pickle=True)
    #     print(
    #         "[INFO] <GetCommitMsg> Save "
    #         + str(len(msgs))
    #         + " commit messages to "
    #         + tempPath
    #         + f"/{type}_msgs.npy."
    #     )

    return msgs


def GetMsgVocab(msgs, type):
    """
    Get the vocabulary of message tokens
    :param msgs - [[[tokens], 0/1], ...]
    :return: vocab - the vocabulary of message tokens. ['tk', 'tk', ...]
             maxLen - the max length of a commit message.
    """

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    fp = open(tempPath + f"/{type}_msglen.csv", "w")

    # get the whole tokens and the max msg length.
    tokens = []
    maxLen = 0

    # for each sample.
    for item in msgs:
        tokens.extend(item[0])
        maxLen = len(item[0]) if (len(item[0]) > maxLen) else maxLen
        fp.write(str(len(item[0])) + "\n")
    fp.close()

    # remove duplicates and get vocabulary.
    vocab = {}.fromkeys(tokens)
    vocab = list(vocab.keys())

    # print.
    print(
        "[INFO] <GetMsgVocab> There are "
        + str(len(vocab))
        + " commit message vocabulary tokens. (except '<pad>')"
    )
    print(
        "[INFO] <GetMsgVocab> The max msg length is "
        + str(maxLen)
        + " tokens. (hyperparameter: _MsgMaxLen_ = "
        + str(_MsgMaxLen_)
        + ")"
    )

    return vocab, maxLen


def GetMsgDict(vocab):
    """
    Get the dictionary of msg vocabulary.
    :param vocab: the vocabulary of msg tokens. ['tk', 'tk', ...]
    :return: tokenDict - the dictionary of msg vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    """

    # get token dict from vocabulary.
    tokenDict = {token: (index + 1) for index, token in enumerate(vocab)}
    tokenDict["<pad>"] = 0

    # print.
    print(
        "[INFO] <GetMsgDict> Create dictionary for "
        + str(len(tokenDict))
        + " msg vocabulary tokens. (with '<pad>')"
    )

    return tokenDict


def GetMsgEmbed(tokenDict, embedSize, type):
    """
    Get the pre-trained weights for embedding layer from the dictionary of msg vocabulary.
    :param tokenDict: the dictionary of msg vocabulary.
    {'tk': 0, 'tk': 1, ..., '<pad>': N}
    :param embedSize: the dimension of the embedding vector.
    :return: preWeights - the pre-trained weights for embedding layer.
    [[n, ...], [n, ...], ...]
    """

    # number of the vocabulary tokens.
    numTokens = len(tokenDict)

    # initialize the pre-trained weights for embedding layer.
    preWeights = np.zeros((numTokens, embedSize))
    for index in range(numTokens):
        preWeights[index] = np.random.normal(size=(embedSize,))
    print(
        "[INFO] <GetMsgEmbed> Create pre-trained embedding weights with "
        + str(len(preWeights))
        + " * "
        + str(len(preWeights[0]))
        + " matrix."
    )

    # save preWeights.
    if not os.path.exists(tempPath + f"/{type}_msgPreWeights.npy"):
        np.save(tempPath + f"/{type}_msgPreWeights.npy", preWeights, allow_pickle=True)
        print(
            "[INFO] <GetMsgEmbed> Save the pre-trained weights of embedding layer to "
            + tempPath
            + f"/{type}_msgPreWeights.npy."
        )

    return preWeights


def GetMsgMapping(msgs, maxLen, tokenDict, type):
    """
    Map the feature data into indexed data.
    :param props: the features of commit messages.
    [[[tokens], 0/1], ...]
    :param maxLen: the max length of the commit message.
    :param tokenDict: the dictionary of commit message vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    :return: np.array(data) - feature data.
             [[n, ...], ...]
             np.array(labels) - labels.
             [[0/1], ...]
    """

    def PadList(dList, pad, length):
        """
        Pad the list data to a fixed length.
        :param dList: the list data - [ , , ...]
        :param pad: the variable used to pad.
        :param length: the fixed length.
        :return: dList - padded list data. [ , , ...]
        """

        if len(dList) <= length:
            dList.extend(pad for i in range(length - len(dList)))
        elif len(dList) > length:
            dList = dList[0:length]

        return dList

    # initialize the data and labels.
    data = []
    labels = []

    # for each sample.
    for item in msgs:
        # process tokens.
        tokens = item[0]
        tokens = PadList(tokens, "<pad>", maxLen)
        # convert tokens into numbers.
        tokens2index = []
        for tk in tokens:
            tokens2index.append(tokenDict[tk])
        data.append(tokens2index)
        # process label.
        label = item[1]
        labels.append([label])

    if _DEBUG_:
        print("[DEBUG] data:")
        print(data[0:3])
        print("[DEBUG] labels:")
        print(labels[0:3])

    # print.
    print(
        "[INFO] <GetMsgMapping> Create "
        + str(len(data))
        + " feature data with 1 * "
        + str(len(data[0]))
        + " vector."
    )
    print(
        "[INFO] <GetMsgMapping> Create "
        + str(len(labels))
        + " labels with 1 * 1 matrix."
    )

    # save files.
    if (not os.path.exists(tempPath + f"/{type}_mdata_" + str(maxLen) + ".npy")) | (
        not os.path.exists(tempPath + f"/{type}_mlabels_" + str(maxLen) + ".npy")
    ):
        np.save(
            tempPath + f"/{type}_mdata_" + str(maxLen) + ".npy", data, allow_pickle=True
        )
        print(
            "[INFO] <GetMsgMapping> Save the mapped numpy data to "
            + tempPath
            + f"/{type}_mdata_"
            + str(maxLen)
            + ".npy."
        )
        np.save(
            tempPath + f"/{type}_mlabels_" + str(maxLen) + ".npy",
            labels,
            allow_pickle=True,
        )
        print(
            "[INFO] <GetMsgMapping> Save the mapped numpy labels to "
            + tempPath
            + f"/{type}_mlabels_"
            + str(maxLen)
            + ".npy."
        )

    return np.array(data), np.array(labels)


class MsgRNN(nn.Module):
    """
    MsgRNN : convert a commit message into a predicted label.
    """

    def __init__(self, preWeights, hiddenSize=32, hiddenLayers=1):
        """
        define each layer in the network model.
        :param preWeights: tensor pre-trained weights for embedding layer.
        :param hiddenSize: node number in the hidden layer.
        :param hiddenLayers: number of hidden layer.
        """

        super(MsgRNN, self).__init__()
        # parameters.
        class_num = 2
        vocabSize, embedDim = preWeights.size()
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocabSize, embedding_dim=embedDim)
        self.embedding.load_state_dict({"weight": preWeights})
        self.embedding.weight.requires_grad = True
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedDim,
            hidden_size=hiddenSize,
            num_layers=hiddenLayers,
            bidirectional=True,
        )
        # Fully-Connected Layer
        self.fc = nn.Linear(hiddenSize * hiddenLayers * 2, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * diff_length * 1.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        """

        # x             batch_size * diff_length * 1
        embeds = self.embedding(x)
        # embeds        batch_size * diff_length * embedding_dim
        inputs = embeds.permute(1, 0, 2)
        # inputs        diff_length * batch_size * (embedding_dim + _DiffExtraDim_)
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        # lstm_out      diff_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # feature_map   batch_size * (hidden_size * num_layers * direction_num)
        final_out = self.fc(feature_map)  # batch_size * class_num
        return self.softmax(final_out)  # batch_size * class_num


def MsgRNNTrain(
    dTrain,
    lTrain,
    dValid,
    lValid,
    preWeights,
    batchsize=64,
    learnRate=0.001,
    dTest=None,
    lTest=None,
):
    """
    Train the MsgRNN model.
    :param dTrain: training data. [[n, ...], ...]
    :param lTrain: training label. [[n, ...], ...]
    :param dValid: validation data. [[n, ...], ...]
    :param lValid: validation label. [[n, ...], ...]
    :param preWeights: pre-trained weights for embedding layer.
    :param batchsize: number of samples in a batch.
    :param learnRate: learning rate.
    :param dTest: test data. [[n, ...], ...]
    :param lTest: test label. [[n, ...], ...]
    :return: model - the MsgRNN model.
    """

    # get the mark of the test dataset.
    if dTest is None:
        dTest = []
    if lTest is None:
        lTest = []
    markTest = 1 if (len(dTest)) & (len(lTest)) else 0

    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    if markTest:
        xTest = torch.from_numpy(dTest).long().cuda()
        yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)
    if markTest:
        test = torchdata.TensorDataset(xTest, yTest)
        testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # get training weights.
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(2):
        weights.append(1 - lbTrain.count(lb) / len(lbTrain))
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWeights = torch.from_numpy(preWeights)
    model = MsgRNN(preWeights, hiddenSize=_MRnnHidSiz_, hiddenLayers=_MRnnHidLay_)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(
        "[INFO] <MsgRNNTrain> ModelType: MsgRNN, HiddenNodes: %d, HiddenLayers: %d."
        % (_MRnnHidSiz_, _MRnnHidLay_)
    )
    print(
        "[INFO] <MsgRNNTrain> BatchSize: %d, LearningRate: %.4f, MaxEpoch: %d, PerEpoch: %d."
        % (batchsize, learnRate, _MRnnMaxEpoch_, _MRnnPerEpoch_)
    )
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(_MRnnMaxEpoch_):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # back propagation.
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                # data conversion.
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                # forward propagation.
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # testing phase.
        if markTest:
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for iter, (data, label) in enumerate(testloader):
                    # data conversion.
                    data = data.to(device)
                    label = label.contiguous().view(-1)
                    label = label.to(device)
                    # forward propagation.
                    yhat = model.forward(data)  # get output
                    # statistic
                    preds = yhat.max(1)[1]
                    predictions.extend(preds.int().tolist())
                    labels.extend(label.int().tolist())
                    torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            # test accuracy.
            accTest = accuracy_score(labels, predictions) * 100

        # output information.
        if 0 == (epoch + 1) % _MRnnPerEpoch_:
            strAcc = "[Epoch {:03}] loss: {:.3}, train acc: {:.3f}%, valid acc: {:.3f}%.".format(
                epoch + 1, lossTrain, accTrain, accValid
            )
            if markTest:
                strAcc = strAcc[:-1] + ", test acc: {:.3f}%.".format(accTest)
            print(strAcc)
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + "/model_MsgRNN.pth")
        # stop judgement.
        if (epoch >= _MRnnJudEpoch_) and (
            accList[-1] < min(accList[-1 - _MRnnJudEpoch_ : -1])
        ):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + "/model_MsgRNN.pth"))
    print(
        "[INFO] <MsgRNNTrain> Finish training MsgRNN model. (Best model: "
        + tempPath
        + "/model_MsgRNN.pth)"
    )

    return model


def MsgRNNTest(model, dTest, lTest, batchsize=64):
    """
    Test the MsgRNN model.
    :param model: deep learning model.
    :param dTest: test data.
    :param lTest: test label.
    :param batchsize: number of samples in a batch
    :return: predictions - predicted labels. [[0], [1], ...]
             accuracy - the total test accuracy. numeric
    """

    # tensor data processing.
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    return predictions, accuracy


def demoPatch():
    """
    demo program of using both commit message and diff code to identify patches.
    """

    # load data.
    if not os.path.exists(tempPath + "/data.npy"):  # | (not _DEBUG_)
        dataLoaded = ReadData()
    else:
        dataLoaded = np.load(tempPath + "/data.npy", allow_pickle=True)
        print(
            "[INFO] <ReadData> Load "
            + str(len(dataLoaded))
            + " raw data from "
            + tempPath
            + "/data.npy."
        )

    # get the diff file properties.
    if not os.path.exists(tempPath + "/props.npy"):
        diffProps = GetDiffProps(dataLoaded)
    else:
        diffProps = np.load(tempPath + "/props.npy", allow_pickle=True)
        print(
            "[INFO] <GetDiffProps> Load "
            + str(len(diffProps))
            + " diff property data from "
            + tempPath
            + "/props.npy."
        )
    # only maintain the diff parts of the code.
    diffProps = ProcessTokens(diffProps, dType=_DTYP_, cType=_CTYP_)
    # normalize the tokens of identifiers, literals, and comments.
    diffProps = AbstractTokens(diffProps, iType=_NIND_, lType=_NLIT_)
    # get the diff token vocabulary.
    diffVocab, diffMaxLen = GetDiffVocab(diffProps)
    # get the max diff length.
    diffMaxLen = _DiffMaxLen_ if (diffMaxLen > _DiffMaxLen_) else diffMaxLen
    # get the diff token dictionary.
    diffDict = GetDiffDict(diffVocab)
    # get pre-trained weights for embedding layer.
    diffPreWeights = GetDiffEmbed(diffDict, _DiffEmbedDim_)
    # get the mapping for feature data and labels.
    diffData, diffLabels = GetDiffMapping(diffProps, diffMaxLen, diffDict)
    # change the tokentypes into one-hot vector.
    diffData = UpdateTokenTypes(diffData)

    # get the commit messages from data.
    if not os.path.exists(tempPath + "/msgs.npy"):
        commitMsgs = GetCommitMsgs(dataLoaded)
    else:
        commitMsgs = np.load(tempPath + "/msgs.npy", allow_pickle=True)
        print(
            "[INFO] <GetCommitMsg> Load "
            + str(len(commitMsgs))
            + " commit messages from "
            + tempPath
            + "/msgs.npy."
        )
    # get the message token vocabulary.
    msgVocab, msgMaxLen = GetMsgVocab(commitMsgs)
    # get the max msg length.
    msgMaxLen = _MsgMaxLen_ if (msgMaxLen > _MsgMaxLen_) else msgMaxLen
    # get the msg token dictionary.
    msgDict = GetMsgDict(msgVocab)
    # get pre-trained weights for embedding layer.
    msgPreWeights = GetMsgEmbed(msgDict, _MsgEmbedDim_)
    # get the mapping for feature data and labels.
    msgData, msgLabels = GetMsgMapping(commitMsgs, msgMaxLen, msgDict)

    # combine the diff data with the message data.
    data, label = CombinePropsMsgs(diffData, msgData, diffLabels, msgLabels)
    # split data into rest/test dataset.
    dataTrain, labelTrain, dataTest, labelTest = SplitData(
        data, label, "test", rate=0.2
    )
    print(
        "[INFO] <main> Get "
        + str(len(dataTrain))
        + " TRAIN data, "
        + str(len(dataTest))
        + " TEST data. (Total: "
        + str(len(dataTrain) + len(dataTest))
        + ")"
    )

    # PatchRNNTrain
    if (_MODEL_) & (os.path.exists(tempPath + "/model_PatchRNN.pth")):
        preWDiff = torch.from_numpy(diffPreWeights)
        preWMsg = torch.from_numpy(msgPreWeights)
        model = PatchRNN(
            preWDiff,
            preWMsg,
            hidSizDiff=_DRnnHidSiz_,
            hidSizMsg=_MRnnHidSiz_,
            hidLayDiff=_DRnnHidLay_,
            hidLayMsg=_MRnnHidLay_,
        )
        model.load_state_dict(torch.load(tempPath + "/model_PatchRNN.pth"))
    else:
        model = PatchRNNTrain(
            dataTrain,
            labelTrain,
            dataTest,
            labelTest,
            preWDiff=diffPreWeights,
            preWMsg=msgPreWeights,
            batchsize=_PRnnBatchSz_,
            learnRate=_PRnnLearnRt_,
            dTest=dataTest,
            lTest=labelTest,
        )

    # PatchRNNTest
    predictions, accuracy = PatchRNNTest(
        model, dataTest, labelTest, batchsize=_PRnnBatchSz_
    )
    _, confusion = OutputEval(predictions, labelTest, "PatchRNN")

    return


def ProcessTokens(props, dType=1, cType=1):
    """
    only maintain the diff parts of the code.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :param dType: 0 - maintain both diff code and context code.
                  1 - only maintain diff code.
    :param cType: 0 - maintain both the code and comments.
                  1 - only maintain code and delete comments.
    :return: props - the normalized features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    """

    # process diff code.
    if 1 == dType:
        propsNew = []
        for item in props:
            # the number of tokens.
            numTokens = len(item[1])
            # item[0]: tokens, item[1]: tokenTypes, item[2]: diffTypes, item[3]: label.
            tokens = [item[0][n] for n in range(numTokens) if (item[2][n])]
            tokenTypes = [item[1][n] for n in range(numTokens) if (item[2][n])]
            diffTypes = [item[2][n] for n in range(numTokens) if (item[2][n])]
            label = item[3]
            # reconstruct sample.
            sample = [tokens, tokenTypes, diffTypes, label]
            propsNew.append(sample)
        props = propsNew
        print("[INFO] <ProcessTokens> Only maintain the diff parts of the code.")

    # process comments.
    if 1 == cType:
        propsNew = []
        for item in props:
            # the number of tokens.
            numTokens = len(item[1])
            # item[0]: tokens, item[1]: tokenTypes, item[2]: diffTypes, item[3]: label.
            tokens = [item[0][n] for n in range(numTokens) if (item[1][n] < 5)]
            tokenTypes = [item[1][n] for n in range(numTokens) if (item[1][n] < 5)]
            diffTypes = [item[2][n] for n in range(numTokens) if (item[1][n] < 5)]
            label = item[3]
            # reconstruct sample.
            sample = [tokens, tokenTypes, diffTypes, label]
            propsNew.append(sample)
        props = propsNew
        print("[INFO] <ProcessTokens> Delete the comment parts of the diff code.")

    # print(props[0])

    return props


def AbstractTokens(props, iType=1, lType=1):
    """
    abstract the tokens of identifiers, literals, and comments.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :param iType:   -1 - not abstract tokens.
                     0 - only abstract variable type and function type. VAR / FUNC
                     1 - abstract the identical variable names and function names.  VAR0, VAR1, ... / FUNC0, FUNC1, ...
    :param lType:   -1 - not abstract tokens.
                     0 - abstract literals with LITERAL.
                     1 - abstract literals with LITERAL/n.
    :return: props - the abstracted features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    """

    if (iType not in [0, 1]) or (lType not in [0, 1]):
        print(
            "[INFO] <AbstractTokens> Not abstract the tokens of identifiers, literals, and comments."
        )
        return props

    for item in props:
        # get tokens and token types.
        tokens = item[0]
        tokenTypes = item[1]
        numTokens = len(tokenTypes)
        # print(tokens)
        # print(tokenTypes)
        # print(numTokens)

        # abstract literals and comments, and separate identifiers into variables and functions.
        markVar = list(np.zeros(numTokens, dtype=int))
        markFuc = list(np.zeros(numTokens, dtype=int))
        for n in range(numTokens):
            # 2: IDENTIFIER, 3: LITERAL, 5: COMMENT
            if 5 == tokenTypes[n]:
                tokens[n] = "COMMENT"
            elif 3 == tokenTypes[n]:
                if 0 == lType:
                    tokens[n] = "LITERAL"
                elif 1 == lType:
                    if not tokens[n].isdigit():
                        tokens[n] = "LITERAL"
            elif 2 == tokenTypes[n]:
                # separate variable name and function name.
                if n < numTokens - 1:
                    if tokens[n + 1] == "(":
                        markFuc[n] = 1
                    else:
                        markVar[n] = 1
                else:
                    markVar[n] = 1
        # print(tokens)
        # print(markVar)
        # print(markFuc)

        # abstract variables and functions.
        if 0 == iType:
            for n in range(numTokens):
                if 1 == markVar[n]:
                    tokens[n] = "VAR"
                elif 1 == markFuc[n]:
                    tokens[n] = "FUNC"
        elif 1 == iType:
            # get variable dictionary.
            varList = [tokens[idx] for idx, mark in enumerate(markVar) if mark == 1]
            varVoc = {}.fromkeys(varList)
            varVoc = list(varVoc.keys())
            varDict = {tk: "VAR" + str(idx) for idx, tk in enumerate(varVoc)}
            # get function dictionary.
            fucList = [tokens[idx] for idx, mark in enumerate(markFuc) if mark == 1]
            fucVoc = {}.fromkeys(fucList)
            fucVoc = list(fucVoc.keys())
            fucDict = {tk: "FUNC" + str(idx) for idx, tk in enumerate(fucVoc)}
            # print(varDict)
            # print(fucDict)
            for n in range(numTokens):
                if 1 == markVar[n]:
                    tokens[n] = varDict[tokens[n]]
                elif 1 == markFuc[n]:
                    tokens[n] = fucDict[tokens[n]]
    # print(tokens)
    print(
        "[INFO] <AbstractTokens> Abstract the tokens of identifiers with iType "
        + str(iType),
        end="",
    )
    print(" (VAR/FUNC).") if (0 == iType) else print(" (VARn/FUNCn).")
    print(
        "[INFO] <AbstractTokens> Abstract the tokens of literals, and comments with iType "
        + str(lType),
        end="",
    )
    print(" (LITERAL/COMMENT).") if (0 == lType) else print(" (LITERAL/n/COMMENT).")

    return props


def CombinePropsMsgs(props, msgs, plabels, mlabels):
    """
    Combine the diff props with the commit messages.
    :param props: diff data. [[[n, {0~5}, {-1~1}], ...], ...] or [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param msgs: message data. [[n, ...], ...]
    :param plabels: diff labels. [[0/1], ...]
    :param mlabels: message labels. [[0/1], ...]
    :return: np.array(data) - combined data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}, n], ...], ...]
             np.array(plabels) - combined labels. [[0/1], ...]
    """

    # check the lengths.
    if len(plabels) != len(mlabels):
        print("[ERROR] <CombinePropsMsgs> the data lengths are mismatch.")
        return [[]], [[]]

    # check the labels.
    cntMatch = 0
    for n in range(len(plabels)):
        if plabels[n][0] == mlabels[n][0]:
            cntMatch += 1
    if cntMatch != len(plabels):
        print(
            "[ERROR] <CombinePropsMsgs> the labels are mismatch. "
            + str(cntMatch)
            + "/"
            + str(len(plabels))
            + "."
        )
        return [[]], [[]]

    # print(props[1], len(props[1]))
    # print(msgs[1], len(msgs[1]))

    data = []
    for n in range(len(plabels)):
        # get the diff prop and message.
        prop = props[n]
        msg = msgs[n]
        # pad data.
        if _DiffMaxLen_ >= _MsgMaxLen_:
            msg = np.pad(msg, (0, _DiffMaxLen_ - _MsgMaxLen_), "constant")
        else:
            prop = np.pad(prop, ((0, _MsgMaxLen_ - _DiffMaxLen_), (0, 0)), "constant")
        # print(msg, len(msg))
        # print(prop, len(prop))
        # reconstruct sample.
        sample = np.vstack((prop.T, msg))
        # append the sample to data.
        data.append(sample.T)

    # print(np.array(data[1]), len(data[1]))
    print("[INFO] <CombinePropsMsgs> Combine the diff props with the commit messages.")

    return np.array(data), np.array(plabels)


class PatchRNN(nn.Module):
    """
    PatchRNN : convert a patch data into a predicted label.
    """

    def __init__(
        self, preWDiff, preWMsg, hidSizDiff=32, hidSizMsg=32, hidLayDiff=1, hidLayMsg=1
    ):
        """
        define each layer in the network model.
        :param preWDiff: tensor pre-trained weights for embedding layer for diff.
        :param preWMsg: tensor pre-trained weights for embedding layer for msg.
        :param hidSizDiff: node number in the hidden layer for diff.
        :param hidSizMsg: node number in the hidden layer for msg.
        :param hidLayDiff: number of hidden layer for diff.
        :param hidLayMsg: number of hidden layer for msg.
        """

        super(PatchRNN, self).__init__()
        # parameters.
        class_num = 2
        # diff.
        vSizDiff, emDimDiff = preWDiff.size()
        # Embedding Layer for diff.
        self.embedDiff = nn.Embedding(num_embeddings=vSizDiff, embedding_dim=emDimDiff)
        self.embedDiff.load_state_dict({"weight": preWDiff})
        self.embedDiff.weight.requires_grad = True
        # LSTM Layer for diff.
        if _DEBUG_:
            print(_DiffExtraDim_)
        self.lstmDiff = nn.LSTM(
            input_size=emDimDiff + _DiffExtraDim_,
            hidden_size=hidSizDiff,
            num_layers=hidLayDiff,
            bidirectional=True,
        )
        # Fully-Connected Layer for diff.
        self.fcDiff = nn.Linear(hidSizDiff * hidLayDiff * 2, hidSizDiff * hidLayDiff)
        # msg.
        vSizMsg, emDimMsg = preWMsg.size()
        # Embedding Layer for msg.
        self.embedMsg = nn.Embedding(num_embeddings=vSizMsg, embedding_dim=emDimMsg)
        self.embedMsg.load_state_dict({"weight": preWMsg})
        self.embedMsg.weight.requires_grad = True
        # LSTM Layer for msg.
        self.lstmMsg = nn.LSTM(
            input_size=emDimMsg,
            hidden_size=hidSizMsg,
            num_layers=hidLayMsg,
            bidirectional=True,
        )
        # Fully-Connected Layer for msg.
        self.fcMsg = nn.Linear(hidSizMsg * hidLayMsg * 2, hidSizMsg * hidLayMsg)
        # common.
        # Fully-Connected Layer.
        self.fc = nn.Linear(
            (hidSizDiff * hidLayDiff + hidSizMsg * hidLayMsg) * 2, class_num
        )
        self.fc1 = nn.Linear(
            (hidSizDiff * hidLayDiff + hidSizMsg * hidLayMsg) * 2,
            hidSizDiff * hidLayDiff + hidSizMsg * hidLayMsg,
        )
        self.fc2 = nn.Linear(hidSizDiff * hidLayDiff + hidSizMsg * hidLayMsg, class_num)
        # Softmax non-linearity.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * diff_length * feature_dim.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        """

        # diff.
        xDiff = x[:, :_DiffMaxLen_, :-1]
        # xDiff         batch_size * diff_length * feature_dim
        # print(xDiff.size())
        embedsDiff = self.embedDiff(xDiff[:, :, 0])
        # embedsDiff    batch_size * diff_length * embed_dim_diff
        features = xDiff[:, :, 1:]
        # features      batch_size * diff_length * _DiffExtraDim_
        inputsDiff = torch.cat((embedsDiff.float(), features.float()), 2)
        # inputsDiff    batch_size * diff_length * (embed_dim_diff + _DiffExtraDim_)
        inputsDiff = inputsDiff.permute(1, 0, 2)
        # inputsDiff    diff_length * batch_size * (embed_dim_diff + _DiffExtraDim_)
        lstm_out, (h_n, c_n) = self.lstmDiff(inputsDiff)
        # lstm_out      diff_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapDiff = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapDiff   batch_size * (hidden_size * num_layers * direction_num)
        # print(featMapDiff.size())
        # msg.
        xMsg = x[:, :_MsgMaxLen_, -1]
        # xMsg          batch_size * msg_length * 1
        # print(xMsg.size())
        embedsMsg = self.embedMsg(xMsg)
        # embedsMsg     batch_size * diff_length * embed_dim_msg
        inputsMsg = embedsMsg.permute(1, 0, 2)
        # inputsMsg     diff_length * batch_size * (embed_dim_msg + _DiffExtraDim_)
        lstm_out, (h_n, c_n) = self.lstmMsg(inputsMsg)
        # lstm_out      diff_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapMsg = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapMsg    batch_size * (hidden_size * num_layers * direction_num)
        # print(featMapMsg.size())
        # common.
        # combine + 1 layer.
        featMap = torch.cat((featMapDiff, featMapMsg), dim=1)
        # print(featMap.size())
        # final_out = self.fc(featMap)        # batch_size * class_num
        # combine + 2 layers.
        featMap = self.fc1(featMap)
        final_out = self.fc2(featMap)
        # separate + 2 layers.
        # featMapDiff = self.fcDiff(featMapDiff)
        # featMapMsg = self.fcMsg(featMapMsg)
        # featMap = torch.cat((featMapDiff, featMapMsg), dim=1)
        # final_out = self.fc2(featMap)
        # print(final_out.size())
        return self.softmax(final_out)  # batch_size * class_num


def PatchRNNTrain(
    dTrain,
    lTrain,
    dValid,
    lValid,
    preWDiff,
    preWMsg,
    batchsize=64,
    learnRate=0.001,
    dTest=None,
    lTest=None,
):
    """
    Train the PatchRNN model.
    :param dTrain: training data. [[n, ...], ...]
    :param lTrain: training label. [[n, ...], ...]
    :param dValid: validation data. [[n, ...], ...]
    :param lValid: validation label. [[n, ...], ...]
    :param preWDiff: pre-trained weights for diff embedding layer.
    :param preWMsg: pre-trained weights for msg embedding layer.
    :param batchsize: number of samples in a batch.
    :param learnRate: learning rate.
    :param dTest: test data. [[n, ...], ...]
    :param lTest: test label. [[n, ...], ...]
    :return: model - the PatchRNN model.
    """

    # get the mark of the test dataset.
    if dTest is None:
        dTest = []
    if lTest is None:
        lTest = []
    markTest = 1 if (len(dTest)) & (len(lTest)) else 0

    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    if markTest:
        xTest = torch.from_numpy(dTest).long().cuda()
        yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)
    if markTest:
        test = torchdata.TensorDataset(xTest, yTest)
        testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # get training weights.
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(2):
        weights.append(1 - lbTrain.count(lb) / len(lbTrain))
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWDiff = torch.from_numpy(preWDiff)
    preWMsg = torch.from_numpy(preWMsg)
    model = PatchRNN(
        preWDiff,
        preWMsg,
        hidSizDiff=_DRnnHidSiz_,
        hidSizMsg=_MRnnHidSiz_,
        hidLayDiff=_DRnnHidLay_,
        hidLayMsg=_MRnnHidLay_,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[INFO] <PatchRNNTrain> ModelType: PatchRNN.")
    print(
        "[INFO] <PatchRNNTrain> Diff Part: EmbedDim: %d, MaxLen: %d, HidNodes: %d, HidLayers: %d."
        % (_DiffEmbedDim_, _DiffMaxLen_, _DRnnHidSiz_, _DRnnHidLay_)
    )
    print(
        "[INFO] <PatchRNNTrain> Msg  Part: EmbedDim: %d, MaxLen: %d, HidNodes: %d, HidLayers: %d."
        % (_MsgEmbedDim_, _MsgMaxLen_, _MRnnHidSiz_, _MRnnHidLay_)
    )
    print(
        "[INFO] <PatchRNNTrain> BatchSize: %d, LearningRate: %.4f, MaxEpoch: %d, PerEpoch: %d, JudEpoch: %d."
        % (batchsize, learnRate, _PRnnMaxEpoch_, _PRnnPerEpoch_, _PRnnJudEpoch_)
    )
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(_PRnnMaxEpoch_):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # back propagation.
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                # data conversion.
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                # forward propagation.
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # testing phase.
        if markTest:
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for iter, (data, label) in enumerate(testloader):
                    # data conversion.
                    data = data.to(device)
                    label = label.contiguous().view(-1)
                    label = label.to(device)
                    # forward propagation.
                    yhat = model.forward(data)  # get output
                    # statistic
                    preds = yhat.max(1)[1]
                    predictions.extend(preds.int().tolist())
                    labels.extend(label.int().tolist())
                    torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            # test accuracy.
            accTest = accuracy_score(labels, predictions) * 100

        # output information.
        if 0 == (epoch + 1) % _PRnnPerEpoch_:
            strAcc = "[Epoch {:03}] loss: {:.3}, train acc: {:.3f}%, valid acc: {:.3f}%.".format(
                epoch + 1, lossTrain, accTrain, accValid
            )
            if markTest:
                strAcc = strAcc[:-1] + ", test acc: {:.3f}%.".format(accTest)
            print(strAcc)
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + "/model_PatchRNN.pth")
        # stop judgement.
        if (epoch >= _PRnnJudEpoch_) and (
            accList[-1] < min(accList[-1 - _PRnnJudEpoch_ : -1])
        ):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + "/model_PatchRNN.pth"))
    print(
        "[INFO] <PatchRNNTrain> Finish training PatchRNN model. (Best model: "
        + tempPath
        + "/model_PatchRNN.pth)"
    )

    return model


def PatchRNNTest(model, dTest, lTest, batchsize=64):
    """
    Test the PatchRNN model.
    :param model: deep learning model.
    :param dTest: test data.
    :param lTest: test label.
    :param batchsize: number of samples in a batch
    :return: predictions - predicted labels. [[0], [1], ...]
             accuracy - the total test accuracy. numeric
    """

    # tensor data processing.
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    return predictions, accuracy


def processProps(diffProps, dataLoaded, type):
    # maintain both the context and diff parts. Delete comments.
    diffProps = ProcessTokens(diffProps, dType=0, cType=_CTYP_)
    # normalize the tokens of identifiers (VARn/FUNCn), literals (LITERAL/n), and comments (none).
    diffProps = AbstractTokens(diffProps, iType=_NIND_, lType=_NLIT_)
    # get the diff token vocabulary.
    diffVocab, _ = GetDiffVocab(diffProps, type)
    # get the diff token dictionary.
    diffDict = GetDiffDict(diffVocab)
    # get pre-trained weights for embedding layer.
    twinPreWeights = GetDiffEmbed(diffDict, _TwinEmbedDim_, type)
    # divide diff code into before/after-version code.
    twinProps, twinMaxLen = DivideBeforeAfter(diffProps, type)
    # get the max twin length.
    twinMaxLen = _TwinMaxLen_ if (twinMaxLen > _DiffMaxLen_) else twinMaxLen
    # get the mapping for feature data and labels.
    twinData, twinLabels = GetTwinMapping(twinProps, twinMaxLen, diffDict, type)
    # change the tokentypes into one-hot vector.
    twinData = UpdateTwinTokenTypes(twinData, type)

    # get the commit messages from data.
    # if not os.path.exists(tempPath + "/msgs.npy"):
    commitMsgs = GetCommitMsgs(dataLoaded, type)
    # else:
    #     commitMsgs = np.load(tempPath + "/msgs.npy", allow_pickle=True)
    #     print(
    #         "[INFO] <GetCommitMsg> Load "
    #         + str(len(commitMsgs))
    #         + " commit messages from "
    #         + tempPath
    #         + "/msgs.npy."
    #     )
    # get the message token vocabulary.
    msgVocab, msgMaxLen = GetMsgVocab(commitMsgs, type)
    # get the max msg length.
    msgMaxLen = _MsgMaxLen_ if (msgMaxLen > _MsgMaxLen_) else msgMaxLen
    # get the msg token dictionary.
    msgDict = GetMsgDict(msgVocab)
    # get pre-trained weights for embedding layer.
    msgPreWeights = GetMsgEmbed(msgDict, _MsgEmbedDim_, type)
    # get the mapping for feature data and labels.
    msgData, msgLabels = GetMsgMapping(commitMsgs, msgMaxLen, msgDict, type)

    # combine the twin data with the message data.
    data, label = CombineTwinMsgs(twinData, msgData, twinLabels, msgLabels)
    return data, label


# 合并权重文件
def merge_twinPreWeights(file_paths):

    # 加载包含权重关键字的文件
    files = []
    for root, _, filenames in os.walk(file_paths):
        for filename in filenames:
            if "twinPreWeights" in filename:
                files.append(os.path.join(root, filename))
    if not files:
        print("No files containing weights found.")
        return

    # 加载权重文件
    weights = [np.load(file_path) for file_path in files]

    # 合并权重矩阵
    merged_twinPreWeights = np.concatenate(weights, axis=0)

    print("[INFO] Weights merged and saved successfully!")
    return merged_twinPreWeights


# 合并权重文件
def merge_msgPreWeights(file_paths):

    # 加载包含权重关键字的文件
    files = []
    for root, _, filenames in os.walk(file_paths):
        for filename in filenames:
            if "msgPreWeights" in filename:
                files.append(os.path.join(root, filename))
    if not files:
        print("No files containing weights found.")
        return

    # 加载权重文件
    weights = [np.load(file_path) for file_path in files]

    # 合并权重矩阵
    merged_msgPreWeights = np.concatenate(weights, axis=0)

    print("[INFO] Weights merged and saved successfully!")
    return merged_msgPreWeights


def demoTwin():
    """
    demo program of using both commit message and diff code to identify patches.
    """

    # load data.
    # if not os.path.exists(tempPath + "/data.npy"):  # | (not _DEBUG_)
    # dataLoaded = ReadData()
    train, test, val = ReadData()
    # test= ReadData()
    # else:
    #     dataLoaded = np.load(tempPath + "/data.npy", allow_pickle=True)
    #     print(
    #         "[INFO] <ReadData> Load "
    #         + str(len(dataLoaded))
    #         + " raw data from "
    #         + tempPath
    #         + "/data.npy."
    #     )

    # get the diff file properties.
    # if not os.path.exists(tempPath + "/props.npy"):
    # diffProps = GetDiffProps(dataLoaded)

    train_diffProps = GetDiffProps(train, "train")
    test_diffProps = GetDiffProps(test, "test")
    val_diffProps = GetDiffProps(val, "val")

    # else:
    #     diffProps = np.load(tempPath + "/props.npy", allow_pickle=True)
    #     print(
    #         "[INFO] <GetDiffProps> Load "
    #         + str(len(diffProps))
    #         + " diff property data from "
    #         + tempPath
    #         + "/props.npy."
    #     )

    dataTrain, labelTrain = processProps(train_diffProps, train, "train")
    dataTest, labelTest = processProps(test_diffProps, test, "test")
    dataValid, labelValid = processProps(val_diffProps, val, "val")

    # split data into rest/test dataset.
    # dataTrain, labelTrain, dataTest, labelTest = SplitData(
    #     data, label, "test", rate=0.2
    # )
    # print(
    #     "[INFO] <demoTwin> Get "
    #     + str(len(dataTrain))
    #     + " TRAIN data, "
    #     + str(len(dataTest))
    #     + " TEST data. (Total: "
    #     + str(len(dataTrain) + len(dataTest))
    #     + ")"
    # )

    twinPreWeights = merge_twinPreWeights(tempPath)
    msgPreWeights = merge_msgPreWeights(tempPath)

    # TwinRNNTrain
    # if (_MODEL_) & (os.path.exists(tempPath + "/model_TwinRNN.pth")):
    #     preWTwin = torch.from_numpy(twinPreWeights)
    #     preWMsg = torch.from_numpy(msgPreWeights)
    #     model = TwinRNN(
    #         preWTwin,
    #         preWMsg,
    #         hidSizTwin=_TRnnHidSiz_,
    #         hidSizMsg=_MRnnHidSiz_,
    #         hidLayTwin=_TRnnHidLay_,
    #         hidLayMsg=_MRnnHidLay_,
    #     )
    #     model.load_state_dict(torch.load(tempPath + "/model_TwinRNN.pth"))
    # # else:
    model = TwinRNNTrain(
        dataTrain,
        labelTrain,
        dataValid,
        labelValid,
        preWTwin=twinPreWeights,
        preWMsg=msgPreWeights,
        batchsize=_TRnnBatchSz_,
        learnRate=_TRnnLearnRt_,
        dTest=dataTest,
        lTest=labelTest,
    )

    # TwinRNNTest
    predictions, accuracy = TwinRNNTest(
        model, dataTest, labelTest, batchsize=_TRnnBatchSz_
    )
    _, confusion = OutputEval(predictions, labelTest, "TwinRNN")

    print(dataset)
    return


def DivideBeforeAfter(diffProps, type):

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    fp = open(tempPath + f"/{type}_twinlen.csv", "w")

    twinProps = []
    maxLen = 0
    # for each sample in diffProps.
    for item in diffProps:
        # get the tk, tkT, dfT, lb.
        tokens = item[0]
        tokenTypes = item[1]
        diffTypes = item[2]
        label = item[3]
        numTokens = len(diffTypes)
        # reconstruct tkB, tkTB, tkA, tkTA.
        tokensB = [tokens[i] for i in range(numTokens) if (diffTypes[i] <= 0)]
        tokenTypesB = [tokenTypes[i] for i in range(numTokens) if (diffTypes[i] <= 0)]
        tokensA = [tokens[i] for i in range(numTokens) if (diffTypes[i] >= 0)]
        tokenTypesA = [tokenTypes[i] for i in range(numTokens) if (diffTypes[i] >= 0)]
        # reconstruct new sample.
        sample = [tokensB, tokenTypesB, tokensA, tokenTypesA, label]
        twinProps.append(sample)
        # get max length.
        maxLenAB = max(len(tokenTypesB), len(tokenTypesA))
        maxLen = maxLenAB if (maxLen < maxLenAB) else maxLen
        fp.write(str(len(tokenTypesB)) + "\n")
        fp.write(str(len(tokenTypesA)) + "\n")
    fp.close()

    # print(twinProps[0])
    # print(maxLen)

    # print.
    print(
        "[INFO] <DivideBeforeAfter> Divide diff code into BEFORE-version and AFTER-version code."
    )
    print(
        "[INFO] <DivideBeforeAfter> The max length in BEFORE/AFTER-version code is "
        + str(maxLen)
        + " tokens. (hyperparameter: _TwinMaxLen_ = "
        + str(_TwinMaxLen_)
        + ")"
    )

    return twinProps, maxLen


def GetTwinMapping(props, maxLen, tokenDict, type):
    """
    Map the feature data into indexed data.
    :param props: the features of diff code.
    [[[tokens], [nums], [tokens], [nums], 0/1], ...]
    :param maxLen: the max length of a twin code.
    :param tokenDict: the dictionary of diff vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    :return: np.array(data) - feature data.
             [[[n, {0~5}, n, {0~5}], ...], ...]
             np.array(labels) - labels.
             [[0/1], ...]
    """

    def PadList(dList, pad, length):
        """
        Pad the list data to a fixed length.
        :param dList: the list data - [ , , ...]
        :param pad: the variable used to pad.
        :param length: the fixed length.
        :return: dList - padded list data. [ , , ...]
        """

        if len(dList) <= length:
            dList.extend(pad for i in range(length - len(dList)))
        elif len(dList) > length:
            dList = dList[0:length]

        return dList

    # initialize the data and labels.
    data = []
    labels = []

    # for each sample.
    for item in props:
        # initialize sample.
        sample = []

        # process tokensB.
        tokens = item[0]
        tokens = PadList(tokens, "<pad>", maxLen)
        tokens2index = []
        for tk in tokens:
            tokens2index.append(tokenDict[tk])
        sample.append(tokens2index)
        # process tokenTypesB.
        tokenTypes = item[1]
        tokenTypes = PadList(tokenTypes, 0, maxLen)
        sample.append(tokenTypes)
        # process tokensA.
        tokens = item[2]
        tokens = PadList(tokens, "<pad>", maxLen)
        tokens2index = []
        for tk in tokens:
            tokens2index.append(tokenDict[tk])
        sample.append(tokens2index)
        # process tokenTypesA.
        tokenTypes = item[3]
        tokenTypes = PadList(tokenTypes, 0, maxLen)
        sample.append(tokenTypes)

        # process sample.
        sample = np.array(sample).T
        data.append(sample)
        # process label.
        label = item[4]
        labels.append([label])

    if _DEBUG_:
        print("[DEBUG] data:")
        print(data[0:3])
        print("[DEBUG] labels:")
        print(labels[0:3])

    # print.
    print(
        "[INFO] <GetTwinMapping> Create "
        + str(len(data))
        + " feature data with "
        + str(len(data[0]))
        + " * "
        + str(len(data[0][0]))
        + " matrix."
    )
    print(
        "[INFO] <GetTwinMapping> Create "
        + str(len(labels))
        + " labels with 1 * 1 matrix."
    )

    # save files.
    if (not os.path.exists(tempPath + f"/{type}_tdata_" + str(maxLen) + ".npy")) | (
        not os.path.exists(tempPath + f"/{type}_tlabels_" + str(maxLen) + ".npy")
    ):
        np.save(
            tempPath + f"/{type}_tdata_" + str(maxLen) + ".npy", data, allow_pickle=True
        )
        print(
            "[INFO] <GetTwinMapping> Save the mapped numpy data to "
            + tempPath
            + f"/{type}_tdata_"
            + str(maxLen)
            + ".npy."
        )
        np.save(
            tempPath + f"/{type}_tlabels_" + str(maxLen) + ".npy",
            labels,
            allow_pickle=True,
        )
        print(
            "[INFO] <GetTwinMapping> Save the mapped numpy labels to "
            + tempPath
            + f"/{type}_tlabels_"
            + str(maxLen)
            + ".npy."
        )

    return np.array(data), np.array(labels)


def UpdateTwinTokenTypes(data, type):
    """
    Update the token type in the feature data into one-hot vector.
    :param data: feature data. [[[n, {0~5}, n, {0~5},], ...], ...]
    :return: np.array(newData). [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, n, 0/1, 0/1, 0/1, 0/1, 0/1], ...], ...]
    """

    newData = []
    # for each sample.
    for item in data:
        # get the transpose of props.
        itemT = item.T
        # initialize new sample.
        newItem = []
        newItem.append(itemT[0])
        newItem.extend(np.zeros((5, len(item)), dtype=int))
        newItem.append(itemT[2])
        newItem.extend(np.zeros((5, len(item)), dtype=int))
        # assign the new sample.
        for i in range(len(item)):
            tokenType = itemT[1][i]
            if tokenType:
                newItem[tokenType][i] = 1
            tokenType = itemT[3][i]
            if tokenType:
                newItem[tokenType + 6][i] = 1
        # get the transpose of new sample.
        newItem = np.array(newItem).T
        # append new sample.
        newData.append(newItem)

    if _DEBUG_:
        print("[DEBUG] newData:")
        print(newData[0:3])

    # print.
    print(
        "[INFO] <UpdateTwinTokenTypes> Update "
        + str(len(newData))
        + " feature data with "
        + str(len(newData[0]))
        + " * "
        + str(len(newData[0][0]))
        + " matrix."
    )

    # save files.
    if not os.path.exists(
        tempPath + f"/{type}_newtdata_" + str(len(newData[0])) + ".npy"
    ):
        np.save(
            tempPath + f"/{type}_newtdata_" + str(len(newData[0])) + ".npy",
            newData,
            allow_pickle=True,
        )
        print(
            "[INFO] <UpdateTwinTokenTypes> Save the mapped numpy data to "
            + tempPath
            + f"/{type}_newtdata_"
            + str(len(newData[0]))
            + ".npy."
        )

    # change marco.
    global _TwinExtraDim_
    _TwinExtraDim_ = 5

    return np.array(newData)


def CombineTwinMsgs(props, msgs, plabels, mlabels):
    """
    Combine the twin props with the commit messages.
    :param props: twin data. [[[n, {0~5}, n, {0~5}], ...], ...] or [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, n, 0/1, 0/1, 0/1, 0/1, 0/1], ...], ...]
    :param msgs: message data. [[n, ...], ...]
    :param plabels: twin labels. [[0/1], ...]
    :param mlabels: message labels. [[0/1], ...]
    :return: np.array(data) - combined data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, n, 0/1, 0/1, 0/1, 0/1, 0/1, n], ...], ...]
             np.array(plabels) - combined labels. [[0/1], ...]
    """

    # check the lengths.
    if len(plabels) != len(mlabels):
        print("[ERROR] <CombineTwinMsgs> the data lengths are mismatch.")
        return [[]], [[]]

    # check the labels.
    cntMatch = 0
    for n in range(len(plabels)):
        if plabels[n][0] == mlabels[n][0]:
            cntMatch += 1
    if cntMatch != len(plabels):
        print(
            "[ERROR] <CombineTwinMsgs> the labels are mismatch. "
            + str(cntMatch)
            + "/"
            + str(len(plabels))
            + "."
        )
        return [[]], [[]]

    # print(props[1], len(props[1]))
    # print(msgs[1], len(msgs[1]))

    data = []
    for n in range(len(plabels)):
        # get the twin prop and message.
        prop = props[n]
        msg = msgs[n]
        print("Shape of prop:", prop.shape)
        print("Shape of msg:", msg.shape)
        # pad data.
        if _TwinMaxLen_ >= _MsgMaxLen_:
            prop = np.pad(prop, ((0, _TwinMaxLen_ - prop.shape[0]), (0, 0)), "constant")
            msg = np.pad(msg, (0, _TwinMaxLen_ - msg.shape[0]), "constant")
        else:
            prop = np.pad(prop, ((0, _MsgMaxLen_ - _TwinMaxLen_), (0, 0)), "constant")
        # print(msg, len(msg))
        # print(prop, len(prop))
        # reconstruct sample.
        print("Shape of arr1:", prop.shape)
        print("Shape of arr2:", msg.shape)
        msg_reshaped = msg.reshape(1, -1)
        print("Shape of arr2:", msg_reshaped.shape)
        sample = np.vstack((prop.T, msg))
        # append the sample to data.
        data.append(sample.T)

    if _DEBUG_:
        print(np.array(data[0:3]))

    print("[INFO] <CombineTwinMsgs> Combine the twin props with the commit messages.")

    return np.array(data), np.array(plabels)


class TwinRNN(nn.Module):
    """
    TwinRNN : convert a patch data into a predicted label.
    """

    def __init__(
        self, preWTwin, preWMsg, hidSizTwin=32, hidSizMsg=32, hidLayTwin=1, hidLayMsg=1
    ):
        """
        define each layer in the network model.
        :param preWTwin: tensor pre-trained weights for embedding layer for twin.
        :param preWMsg: tensor pre-trained weights for embedding layer for msg.
        :param hidSizTwin: node number in the hidden layer for twin.
        :param hidSizMsg: node number in the hidden layer for msg.
        :param hidLayTwin: number of hidden layer for twin.
        :param hidLayMsg: number of hidden layer for msg.
        """

        super(TwinRNN, self).__init__()
        # parameters.
        class_num = 2
        # twin.
        vSizTwin, emDimTwin = preWTwin.size()
        # Embedding Layer for twin.
        self.embedTwin = nn.Embedding(num_embeddings=vSizTwin, embedding_dim=emDimTwin)
        self.embedTwin.load_state_dict({"weight": preWTwin})
        self.embedTwin.weight.requires_grad = True
        # LSTM Layer for twin.
        if _DEBUG_:
            print(_TwinExtraDim_)
        self.lstmTwin = nn.LSTM(
            input_size=emDimTwin + _TwinExtraDim_,
            hidden_size=hidSizTwin,
            num_layers=hidLayTwin,
            bidirectional=True,
        )
        # msg.
        vSizMsg, emDimMsg = preWMsg.size()
        # Embedding Layer for msg.
        self.embedMsg = nn.Embedding(num_embeddings=vSizMsg, embedding_dim=emDimMsg)
        self.embedMsg.load_state_dict({"weight": preWMsg})
        self.embedMsg.weight.requires_grad = True
        # LSTM Layer for msg.
        self.lstmMsg = nn.LSTM(
            input_size=emDimMsg,
            hidden_size=hidSizMsg,
            num_layers=hidLayMsg,
            bidirectional=True,
        )
        # common.
        # Fully-Connected Layer.
        self.fc1 = nn.Linear(hidSizTwin * hidLayTwin * 4, hidSizTwin * hidLayTwin * 2)
        self.fc2 = nn.Linear(hidSizTwin * hidLayTwin * 2, class_num)
        self.fc3 = nn.Linear(
            (hidSizTwin * hidLayTwin + hidSizMsg * hidLayMsg) * 2,
            hidSizTwin * hidLayTwin + hidSizMsg * hidLayMsg,
        )
        self.fc4 = nn.Linear(hidSizTwin * hidLayTwin + hidSizMsg * hidLayMsg, class_num)
        # Softmax non-linearity.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * twin_length * feature_dim.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        """

        # twin 1.
        xTwin = x[:, :_TwinMaxLen_, :6]
        # xTwin         batch_size * twin_length * feature_dim
        # print(xTwin.size())
        embedsTwin = self.embedTwin(xTwin[:, :, 0])
        # embedsTwin    batch_size * twin_length * embed_dim_twin
        features = xTwin[:, :, 1:]
        # features      batch_size * twin_length * _TwinExtraDim_
        inputsTwin = torch.cat((embedsTwin.float(), features.float()), 2)
        # print(inputsTwin.size())
        # inputsTwin    batch_size * twin_length * (embed_dim_twin + _TwinExtraDim_)
        inputsTwin = inputsTwin.permute(1, 0, 2)
        # inputsTwin    twin_length * batch_size * (embed_dim_twin + _TwinExtraDim_)
        lstm_out, (h_n, c_n) = self.lstmTwin(inputsTwin)
        # lstm_out      twin_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapTwin1 = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapTwin1   batch_size * (hidden_size * num_layers * direction_num)
        # print(featMapTwin1)
        # twin 2.
        xTwin = x[:, :_TwinMaxLen_, 6:-1]
        # xTwin         batch_size * twin_length * feature_dim
        # print(xTwin.size())
        embedsTwin = self.embedTwin(xTwin[:, :, 0])
        # embedsTwin    batch_size * twin_length * embed_dim_twin
        features = xTwin[:, :, 1:]
        # features      batch_size * twin_length * _TwinExtraDim_
        inputsTwin = torch.cat((embedsTwin.float(), features.float()), 2)
        # print(inputsTwin.size())
        # inputsTwin    batch_size * twin_length * (embed_dim_twin + _TwinExtraDim_)
        inputsTwin = inputsTwin.permute(1, 0, 2)
        # inputsTwin    twin_length * batch_size * (embed_dim_twin + _TwinExtraDim_)
        lstm_out, (h_n, c_n) = self.lstmTwin(inputsTwin)
        # lstm_out      twin_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapTwin2 = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapTwin2   batch_size * (hidden_size * num_layers * direction_num)
        # print(featMapTwin2)
        # msg.
        xMsg = x[:, :_MsgMaxLen_, -1]
        # xMsg          batch_size * msg_length * 1
        # print(xMsg.size())
        embedsMsg = self.embedMsg(xMsg)
        # embedsMsg     batch_size * msg_length * embed_dim_msg
        inputsMsg = embedsMsg.permute(1, 0, 2)
        # inputsMsg     msg_length * batch_size * (embed_dim_msg)
        lstm_out, (h_n, c_n) = self.lstmMsg(inputsMsg)
        # lstm_out      msg_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapMsg = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapMsg    batch_size * (hidden_size * num_layers * direction_num)
        # print(featMapMsg.size())
        # common.
        # combine twins.
        featMap = torch.cat((featMapTwin1, featMapTwin2), dim=1)
        # fc layers.
        featMap = self.fc1(featMap)
        if 0 == _TWIN_:  # (only twins).
            final_out = self.fc2(featMap)
        elif 1 == _TWIN_:  # (twins + msg).
            # combine twins + msg.
            featMap = torch.cat((featMap, featMapMsg), dim=1)
            # fc 2 layers.
            featMap = self.fc3(featMap)
            final_out = self.fc4(featMap)
        # print(final_out.size())
        return self.softmax(final_out)  # batch_size * class_num


def TwinRNNTrain(
    dTrain,
    lTrain,
    dValid,
    lValid,
    preWTwin,
    preWMsg,
    batchsize=64,
    learnRate=0.001,
    dTest=None,
    lTest=None,
):
    """
    Train the TwinRNN model.
    :param dTrain: training data. [[n, ...], ...]
    :param lTrain: training label. [[n, ...], ...]
    :param dValid: validation data. [[n, ...], ...]
    :param lValid: validation label. [[n, ...], ...]
    :param preWDiff: pre-trained weights for diff embedding layer.
    :param preWMsg: pre-trained weights for msg embedding layer.
    :param batchsize: number of samples in a batch.
    :param learnRate: learning rate.
    :param dTest: test data. [[n, ...], ...]
    :param lTest: test label. [[n, ...], ...]
    :return: model - the TwinRNN model.
    """

    # get the mark of the test dataset.
    if dTest is None:
        dTest = []
    if lTest is None:
        lTest = []
    markTest = 1 if (len(dTest)) & (len(lTest)) else 0

    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    if markTest:
        xTest = torch.from_numpy(dTest).long().cuda()
        yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)
    if markTest:
        test = torchdata.TensorDataset(xTest, yTest)
        testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # get training weights.
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(2):
        weights.append(1 - lbTrain.count(lb) / len(lbTrain))
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWTwin = torch.from_numpy(preWTwin)
    preWMsg = torch.from_numpy(preWMsg)
    model = TwinRNN(
        preWTwin,
        preWMsg,
        hidSizTwin=_TRnnHidSiz_,
        hidSizMsg=_MRnnHidSiz_,
        hidLayTwin=_TRnnHidLay_,
        hidLayMsg=_MRnnHidLay_,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[INFO] <TwinRNNTrain> ModelType: TwinRNN.")
    print(
        "[INFO] <TwinRNNTrain> Code Part: EmbedDim: %d, MaxLen: %d, HidNodes: %d, HidLayers: %d."
        % (_TwinEmbedDim_, _TwinMaxLen_, _TRnnHidSiz_, _TRnnHidLay_)
    )
    print(
        "[INFO] <TwinRNNTrain> Msg  Part: EmbedDim: %d, MaxLen: %d, HidNodes: %d, HidLayers: %d."
        % (_MsgEmbedDim_, _MsgMaxLen_, _MRnnHidSiz_, _MRnnHidLay_)
    )
    print(
        "[INFO] <TwinRNNTrain> BatchSize: %d, LearningRate: %.4f, MaxEpoch: %d, PerEpoch: %d, JudEpoch: %d."
        % (batchsize, learnRate, _TRnnMaxEpoch_, _TRnnPerEpoch_, _TRnnJudEpoch_)
    )
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(_TRnnMaxEpoch_):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # back propagation.
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                # data conversion.
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                # forward propagation.
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # testing phase.
        if markTest:
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for iter, (data, label) in enumerate(testloader):
                    # data conversion.
                    data = data.to(device)
                    label = label.contiguous().view(-1)
                    label = label.to(device)
                    # forward propagation.
                    yhat = model.forward(data)  # get output
                    # statistic
                    preds = yhat.max(1)[1]
                    predictions.extend(preds.int().tolist())
                    labels.extend(label.int().tolist())
                    torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            # test accuracy.
            accTest = accuracy_score(labels, predictions) * 100

        # output information.
        if 0 == (epoch + 1) % _TRnnPerEpoch_:
            strAcc = "[Epoch {:03}] loss: {:.3}, train acc: {:.3f}%, valid acc: {:.3f}%.".format(
                epoch + 1, lossTrain, accTrain, accValid
            )
            if markTest:
                strAcc = strAcc[:-1] + ", test acc: {:.3f}%.".format(accTest)
            print(strAcc)
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + "/model_TwinRNN.pth")
        # stop judgement.
        if (epoch >= _TRnnJudEpoch_) and (
            accList[-1] < min(accList[-1 - _TRnnJudEpoch_ : -1])
        ):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + "/model_TwinRNN.pth"))
    print(
        "[INFO] <TwinRNNTrain> Finish training TwinRNN model. (Best model: "
        + tempPath
        + "/model_TwinRNN.pth)"
    )

    return model


def TwinRNNTest(model, dTest, lTest, batchsize=64):
    """
    Test the TwinRNN model.
    :param model: deep learning model.
    :param dTest: test data.
    :param lTest: test label.
    :param batchsize: number of samples in a batch
    :return: predictions - predicted labels. [[0], [1], ...]
             accuracy - the total test accuracy. numeric
    """

    # tensor data processing.
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    return predictions, accuracy


if __name__ == "__main__":
    start = time.time()
    dataset = "Detect0day/balance/1:99"
    test_val = "Detect0day"
    print(dataset)
    tempPath = tempPath + dataset
    # demoDiffRNN()
    # demoCommitMsg()
    # demoPatch()
    demoTwin()
    end = time.time()
    print(end - start)
    print(dataset)
