'''
    Copy the valid graph log files from '_GraphLogs' to 'data_raw'.
'''

import os
import sys

rootPath = './'
dataPath = './_PatchDB/'            # reference folder.
sampPath = './_GraphLogs/'          # source folder.
destPath = rootPath + '/_data/data_raw/'  # destination folder.
logsPath = rootPath + '/logs/'      # log folder.

logsName = 'out_slim_ninf_noast_n1_w.log'

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

def CopyFile(src, dst):
    # read file.
    try:
        fp = open(src, encoding='utf-8', errors='ignore')
    except:
        print('[ERROR] Cannot open ' + src)
    lines = fp.readlines()
    fp.close()

    # write file.
    fp = open(dst, 'w')
    for line in lines:
        fp.write(line.encode("gbk", 'ignore').decode("gbk", "ignore"))
    fp.close()

    return 0

def main():
    # get positive patch list.
    posPatchList = []
    for _, _, files in os.walk(dataPath + '/security_patch/'):
        posPatchList.extend(files)
    for _, _, files in os.walk(dataPath + '/positives/'):
        posPatchList.extend(files)
    print('[INFO] There are ' + str(len(posPatchList)) + ' positive patches.')
    # get negative patch list.
    negPatchList = []
    for _, _, files in os.walk(dataPath + '/negatives/'):
        negPatchList.extend(files)
    print('[INFO] There are ' + str(len(negPatchList)) + ' negative patches.')
    
    # get sample list.
    logsList = os.listdir(sampPath)
    if '.DS_Store' in logsList: 
        logsList.remove('.DS_Store')
    print('[INFO] There are ' + str(len(logsList)) + ' graph log files.')

    # get log files.
    cnt = 0
    for file in logsList:
        # determine the subfolder.
        if (file in posPatchList): # if file in positive patch list.
            subfolder = 'positives'
        elif (file in negPatchList): # if file in negative patch list.
            subfolder = 'negatives'
        else: # if file is a new patch that is not in the existing data set.
            print('[WARNING] Cannot find file in PatchDB: ' + file)
            continue

        # define src and dst.
        src = os.path.join(sampPath + '/' + file, logsName)
        dst = os.path.join(destPath + '/' + subfolder, file + '.log')

        if os.path.exists(src):  # if find file/out.log.
            cnt += 1
            if os.path.exists(dst): # if already have log file in dst.
                print('[INFO] [' + str(cnt) + '] Already found log file in ' + dst)
            else: # if cannot find log file in dst.
                print('[INFO] [' + str(cnt) + '] Copy log file from ' + src)
                CopyFile(src, dst)
        else:  # if cannot find file/out.log
            print('[WARNING] Cannot find log file ' + src)

    return

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'get_dataset.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # check destination folders.
    if (not os.path.exists(destPath)):
        os.makedirs(destPath)
    if (not os.path.exists(destPath + '/positives/')):
        os.makedirs(destPath + '/positives/')
    if (not os.path.exists(destPath + '/negatives/')):
        os.makedirs(destPath + '/negatives/')
    # main entrance.
    main()
