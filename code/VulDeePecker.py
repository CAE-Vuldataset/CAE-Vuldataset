import os
import sys

program_path = "/home/platform/test/database/VulDeePecker"
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
                if fpath.endswith('.c') or fpath.endswith('.cpp'):
                    if fpath.split('/')[-2].find("CVE")!=-1:
                        year=int(fpath.split('/')[-2].split('-')[1])
                        print(year)
                        if year>=2013:
                            latter=latter+1
                        else:
                            former=former+1
                    #print(fpath)
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
    avg=sum/num
    print("maxnum= ",maxnum," minnum= ",minnum," avg= ",avg)
    list.sort()
    print("mid=",list[len(list)/2])
    print(list2)
    print("former",former,"latter",latter)
