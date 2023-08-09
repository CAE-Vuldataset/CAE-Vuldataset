import os
import sys
import json
maxnum=0
minnum=99999999999
sum=0
num=0
list=[]
list2=[]
program_path = "" # your path

if __name__ == '__main__':
    with open(program_path) as f:
        data=json.load(f)
    for block in data:

        linenum=int(block["size"])
        project=block["project"]
        if project not in list2:
            list2.append(project)
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
    print(list2)