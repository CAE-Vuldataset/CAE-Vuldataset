import sys
import json

def getline(f):
    count = 0
    for index, line in enumerate(f):
        count += 1




path="~/test/database/Lipp_et_al/dataset/"
softwares = ['binutils','ffmpeg','libpng','libtiff','libxml2','openssl','php','poppler','sqlite3']
maxnum=0
minnum=sys.maxint
sum=0
num=0
for software in softwares:
    file = path+software+"/sca_results.json"
    with open(file,'r') as fp:
        json_data = json.load(fp)
        lines=json_data['findings']