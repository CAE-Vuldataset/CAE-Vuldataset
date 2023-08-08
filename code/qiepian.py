import os
import sys

program_path1 = "/home/platform/test/database/VulDeeLocator/data/iSeVCs/iSeVCs_for_train_programs/AD_slices.txt"
program_path2="/home/platform/test/database/VulDeeLocator/data/iSeVCs/iSeVCs_for_train_programs/AE_slices.txt"
program_path3="/home/platform/test/database/VulDeeLocator/data/iSeVCs/iSeVCs_for_train_programs/FC_slices.txt"
program_path4="/home/platform/test/database/VulDeeLocator/data/iSeVCs/iSeVCs_for_train_programs/PD_slices.txt"
maxnum=0
minnum=sys.maxint
sum=0
num=0
list=[]
list2=[]
year1=1990
year2=2024
former=0
latter=0
def getline(fpath):
    global maxnum,minnum,sum,num,list,list2,year1,year2,former,latter
    linenum=0
    with open(fpath,"r") as f:
        for index, line in enumerate(f):
            if(line.find("CVE")!=-1):
                year=int(line[(line.find("CVE")+4):(line.find("CVE")+8)])
                year1=max(year,year1)
                year2=min(year,year2)
                if year>=2016:
                    latter=latter+1
                else:
                    former=former+1
            if(line.find("----")==-1):
                linenum=linenum+1
            else:
                maxnum = max(maxnum, linenum)
                minnum = min(minnum, linenum)
                sum = sum + linenum
                num = num + 1
                list.append(linenum)
                linenum=0



if __name__ == '__main__':
    getline(program_path1)
    getline(program_path2)
    getline(program_path3)
    getline(program_path4)
    avg=sum*1.0/num
    print("maxnum= ",maxnum," minnum= ",minnum," avg= ",avg)
    list.sort()
    print("mid=",list[len(list)/2])
    print(list2)
    print("year2:",year2,"year1 ",year1)