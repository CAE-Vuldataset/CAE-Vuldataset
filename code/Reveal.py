import os
import sys
import json
maxnum=0
minnum=99999999999
sum=0
num=0
list=[]
program_path = "D:\leak\数据集\dataset.json"

if __name__ == '__main__':
    with open(program_path) as f:
        data=json.load(f)
    for block in data:
        if block["target"]==0:
            continue
        func=block['func']
        linenum=func.count("\n")
        if(linenum==1):
            print(num+1)
            print(func)
        list.append(linenum)
        maxnum = max(maxnum, linenum)
        minnum = min(minnum, linenum)
        sum = sum + linenum
        num = num + 1
    avg=sum*1.0/num
    print("maxnum= ",maxnum," minnum= ",minnum," avg= ",avg)
    list.sort()
    print("mid=",list[int(len(list)/2)])
    print("num:",num)