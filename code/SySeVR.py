import os
import sys

program_path = "./SySeVR/Program data/NVD/" # your path
maxnum=0
minnum=sys.maxint
sum=0
num=0
list=[]
list2=[]
former=0
latter=0
def getline(fpath):
    count=0
    with open(fpath,"r") as f:
        count=len(f.readlines())
    return count

def findfile(pathName):
    global maxnum,minnum,sum,num,list,list2,former,latter
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
                if (fpath.endswith('.c') or fpath.endswith('.cpp')) and fpath.find("_PATCHED")==-1 :
                    if fpath.split('/')[-2].find("NVD")!=-1:
                        if len(fpath.split('/')[-1].split('-'))>1:
                            year=int(fpath.split('/')[-1].split('-')[1])
                        else:
                            year=year=int(fpath.split('/')[-1].split('_')[1])
                        print(year)
                        if year>=2016:
                            latter=latter+1
                        else:
                            former=former+1
                    print(fpath)
                    linenum=getline(fpath)
                    if(linenum<20):
                        list2.append(fpath)
                    list.append(linenum)
                    maxnum=max(maxnum,linenum)
                    minnum=min(minnum,linenum)
                    sum=sum+linenum
                    num=num+1

if __name__ == '__main__':
    findfile(program_path)
    if(num):
        avg=sum*1.0/num
    print("maxnum= ",maxnum," minnum= ",minnum," avg= ",avg)
    list.sort()
    print("mid=",list[len(list)/2])
    print(list2)
    print("formmer",former,"latter",latter)
