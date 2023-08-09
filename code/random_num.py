import random
import os

# Take 97 numbers
cnt = 97
# Vulnerability sample storage directory of the vulnerability data set
sample_path = './SySeVR/NVD'
# Get the number of vulnerability samples
samples = os.listdir(sample_path)
num = len(samples)

random_nums = []

# Get 97 random numbers
while(cnt > 0):
    random_number = random.randint(0, num - 1)
    while(random_number not in random_nums): # make sure there are no duplicates
        random_nums.append(random_number)
        print(random_number)
        cnt -= 1

# Take out the samples corresponding to 97 random numbers
for ran in random_nums:
    print(samples[ran])
