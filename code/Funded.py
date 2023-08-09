import os
import sys

program_path = "./SARD/" # your path
maxnum=0
minnum=sys.maxint
sum=0
num=0
list=[]
def getline(fpath):
    count=0
    with open(fpath,"r") as f:
        count=len(f.readlines())
    return count

def findfile(pathName):
    global maxnum,minnum,sum,num,list
    if os.path.exists(pathName):
        fileList = os.listdir(pathName)
        for f in fileList:
            if f == "$RECYCLE.BIN" or f == "System Volume Information":
                continue
            fpath = os.path.join(pathName, f)
            if os.path.isdir(fpath):
                fpath=fpath+"/"
                findfile(fpath)
            else:
                if fpath.endswith('.txt'):
                    print(fpath)
                    linenum=getline(fpath)
                    list.append(linenum)
                    maxnum=max(maxnum,linenum)
                    minnum=min(minnum,linenum)
                    sum=sum+linenum
                    num=num+1

if __name__ == '__main__':
    findfile(program_path)
    avg=sum*1.0/num
    print("maxnum= ",maxnum," minnum= ",minnum," avg= ",avg)
    list.sort()
    print("mid=",list[len(list)/2])
