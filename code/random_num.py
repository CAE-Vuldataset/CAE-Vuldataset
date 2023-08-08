import random
import os

# 取97个数
cnt = 7
# 漏洞数据集的漏洞样本存放目录
sample_path = '/Volumes/.../SySeVR/NVD'
# 获得漏洞样本的数量
samples = os.listdir(sample_path)
num = len(samples)

random_nums = []

# 获得97个随机数
while(cnt > 0):
    random_number = random.randint(0, num - 1)
    while(random_number not in random_nums): # 确保没有重复
        random_nums.append(random_number)
        print(random_number)
        cnt -= 1

# 取出97个随机数对应的样本
for ran in random_nums:
    print(samples[ran])
