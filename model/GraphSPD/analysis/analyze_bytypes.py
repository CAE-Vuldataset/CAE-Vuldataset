import os
import sys
import csv
import pandas as pd

dataPath = './_PatchDB/positives/'
logsPath = './logs/'
expsFile = logsPath + 'TestResult.txt'

# sparse the log file.
df = pd.read_csv(expsFile, usecols=['filename', 'label', 'prediction'], sep=',')
# print(df)

posClass = os.listdir(dataPath)
for cls in posClass:
    posFolder = os.path.join(dataPath, cls)
    files = os.listdir(posFolder)
    cnt = 0
    cnt1 = 0

    for file in files:
        if file in df['filename'].values:
            cnt += 1
            row = df.loc[df['filename'] == file]
            if row['prediction'].values[0] == 1:
                cnt1 += 1

    print(cls, ',', cnt, ',', cnt1)
