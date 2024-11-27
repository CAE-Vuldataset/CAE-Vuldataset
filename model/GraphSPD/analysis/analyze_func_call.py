import os
import numpy as np
from collections import Counter
np.set_printoptions(threshold=np.inf)

rootPath = './'
tempPath = './'
mdatPath = rootPath + '/data_mid/'
foutPath = tempPath + '/logs/'

def main():
    cal_func_calls(mdatPath + '/positives/', 'func_call_pos', 0)
    cal_func_calls(mdatPath + '/negatives/', 'func_call_neg', 0)
    cal_func_calls(mdatPath + '/positives/', 'func_call_pos', 1)
    cal_func_calls(mdatPath + '/negatives/', 'func_call_neg', 1)
    return

def cal_func_calls(datafolder, savefilename, option=0):
    func_call = []
    cnt = 0
    for root, ds, fs in os.walk(datafolder):
        for file in fs:
            # if (cnt >= 1): continue
            if ('.DS_Store' in file): continue
            # =====================================================
            filename = os.path.join(root, file).replace('\\', '/')
            cnt += 1
            print(f'[{cnt}] Read middle file from {filename}.')
            data = np.load(filename, allow_pickle=True)
            nodes = data['nodes']
            temp = []
            for n_nodes in range(len(nodes)):
                node = nodes[n_nodes]
                if (0 == node[1]):
                    continue
                tokenTypes = node[-2]
                tokens = node[-1]
                num = len(tokenTypes)
                for i in range(num - 1):
                    if (2 == tokenTypes[i]) and (2 <= len(tokens[i])) and ('(' == tokens[i + 1]):
                        # print(tokens[i])
                        temp.append(tokens[i])
            if (option):
                temp = list(set(temp))
            func_call.extend(temp)

    call_counter = Counter(func_call)
    call_sorted = sorted(call_counter.items(), key=lambda x: x[1], reverse=True)
    savefile = foutPath + '/' + savefilename + '_' + str(option) + '.csv'
    f = open(savefile, 'w')
    f.write('func_call,count\n')
    for i in range(len(call_sorted)):
        record = call_sorted[i]
        f.write(record[0] + ',' + str(record[1]) + '\n')
    f.close()
    print('[INFO] Save analytics file in ' + savefile)

    return

if __name__ == '__main__':
    main()