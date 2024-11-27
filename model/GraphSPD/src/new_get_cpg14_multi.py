import os
import re
import shutil
import urllib.request
import subprocess

def CheckPaths(opt, commitID):
	[pdifPath, repoPath, abfsPath] = [opt.patch_path, opt.repo_path, opt.ab_path]
	if not os.path.exists(pdifPath):
		os.makedirs(pdifPath)
	if not os.path.exists(abfsPath):
		os.makedirs(abfsPath)
	if not os.path.exists(abfsPath + f"/{commitID}/a/"):
		os.makedirs(abfsPath + f"/{commitID}/a/")
	if not os.path.exists(abfsPath + f"/{commitID}/b/"):
		os.makedirs(abfsPath + f"/{commitID}/b/")
	if not os.path.exists(repoPath):
		os.makedirs(repoPath)
	if not os.path.exists(os.path.join(abfsPath, commitID)):
		os.makedirs(os.path.join(abfsPath, commitID))
	return 0

def GetFilesAB(opt, owner, repo, commitID):
	[pdifPath, repoPath, abfsPath] = [opt.patch_path, opt.repo_path, opt.ab_path]
	CheckPaths(opt, commitID)

	# # find if the url link is avaiable.
	# try: 
	# 	_code_ = urllib.request.urlopen(f"https://github.com/{owner}/{repo}/commit/{commitID}.patch").code
	# except Exception as err:
	# 	print(f'[ERROR] {err} https://github.com/{owner}/{repo}/commit/{commitID}.patch')
	# 	shutil.rmtree(f'{abfsPath}/{commitID}/')
	# 	print(f'[INFO] Removing folder {abfsPath}/{commitID}/')
	# 	return 1

	# # download the commit/patch.
	# urllib.request.urlretrieve(f"https://github.com/{owner}/{repo}/commit/{commitID}.patch", \
	# 		    os.path.join(pdifPath, commitID))
	
	# 构造下载链接
	# 作者的下载方法总是报错HTTP Error 429: Too Many Requests
	# url = f"https://github.com/{owner}/{repo}/commit/{commitID}.patch"
	# output_file = os.path.join(pdifPath, commitID)

	# # 尝试下载文件
	# for attempt in range(3):
	# 	try:
	# 		print(f"Attempting to download {url} (Attempt {attempt + 1})")
	# 		result = subprocess.run(
	# 			["wget", url, "-O", output_file, "--timeout=10", "--tries=1"],
	# 			check=True
	# 		)
	# 		print(f"Downloaded patch to {output_file}")
	# 		break
	# 	except subprocess.CalledProcessError as e:
	# 		print(f"Attempt {attempt + 1} failed for {url}: {e}")

	# 跳过下载，直接生成修复前后的文件
	# read the commit.
	pLines = open(os.path.join(pdifPath, commitID), encoding='utf-8', errors='ignore').readlines()
	# get the AB file list.
	filesAB = []
	pattern = "diff --git a/(.*) b/(.*)"
	for pLine in pLines:
		contents = re.findall(pattern, pLine)
		if 0 == len(contents):
			continue
		filesAB.append(list(contents[0]))

	# get the newest repo.
	if not os.path.exists(os.path.join(repoPath, repo)):
		os.system(f'cd {repoPath}; git clone https://github.com/{owner}/{repo}.git')
	
	# rollback to B, and get the relevant files in B.
	# 如果b文件夹存在，则删除该文件夹并重新创建
	# if os.path.exists(f'{abfsPath}/{commitID}/b/'):
	# 	shutil.rmtree(f'{abfsPath}/{commitID}/b/')
	# 	os.mkdir(f'{abfsPath}/{commitID}/b/')
	# 如果b文件夹为空，才去找修改后的文件
	if os.listdir(f'{abfsPath}/{commitID}/b/') == []:
		os.system(f'cd {repoPath}/{repo}; git reset --hard {commitID}')
		for [_, fileB] in filesAB:
			os.system(f'cp {repoPath}/{repo}/{fileB} {abfsPath}/{commitID}/b/')

	# rollback to A, and get the relevant files in A.
	out = os.popen(f'cd {repoPath}/{repo}; git rev-list --parents -n 1 {commitID}').read()
	commitA = out[out.find(' ')+1:].rstrip()
	# if os.path.exists(f'{abfsPath}/{commitID}/a/'):
	# 	shutil.rmtree(f'{abfsPath}/{commitID}/a/')
	# 	os.mkdir(f'{abfsPath}/{commitID}/a/')
	if os.listdir(f'{abfsPath}/{commitID}/a/') == []:
		os.system(f'cd {repoPath}/{repo}; git reset --hard {commitA}')
		for [fileA, _] in filesAB:
			os.system(f'cp {repoPath}/{repo}/{fileA} {abfsPath}/{commitID}/a/')

	# get brief diff file.
	if os.path.exists(f'{abfsPath}/{commitID}/diff.patch'):
		os.remove(f'{abfsPath}/{commitID}/diff.patch')
	os.system(f'diff -brN -U 0 -p {abfsPath}/{commitID}/a/ {abfsPath}/{commitID}/b/ >> {abfsPath}/{commitID}/diff.patch')

	return 0

def GetCPG14(opt, commitID):
    abfsPath = opt.ab_path
    if abfsPath.startswith('./'):
        abfsPath = abfsPath[2:]

    # output the function name, file, and lines.
    if not os.path.exists(f'{abfsPath}/{commitID}/funcA.txt') and os.listdir(f'{abfsPath}/{commitID}/a/') != []:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        os.system(f'cd ./joern/joern-cli/; ./joern --script ../../src/locateFunc.sc \
                  --param inputFile=../../{abfsPath}/{commitID}/a/ --param outFile=../../{abfsPath}/{commitID}/funcA.txt')
    if not os.path.exists(f'{abfsPath}/{commitID}/funcB.txt') and os.listdir(f'{abfsPath}/{commitID}/b/') != []:
        os.system(f'cd ./joern/joern-cli/; ./joern --script ../../src/locateFunc.sc \
                  --param inputFile=../../{abfsPath}/{commitID}/b/ --param outFile=../../{abfsPath}/{commitID}/funcB.txt')
    
    # output the cpg14 of different functions.
    # if os.listdir(f'{abfsPath}/{commitID}/cpgsA/') == []:
    #     print('##################################################')
    #     shutil.rmtree(f'{abfsPath}/{commitID}/cpgsA/')
    #     os.system(f'cd ./joern/joern-cli; ./joern-parse  ../../{abfsPath}/{commitID}/a/; \
	# 			./joern-export --repr cpg14 --out ../../{abfsPath}/{commitID}/cpgsA/')
    # if os.listdir(f'{abfsPath}/{commitID}/cpgsB/') == []:
    #     shutil.rmtree(f'{abfsPath}/{commitID}/cpgsB/')
    #     os.system(f'cd ./joern/joern-cli; ./joern-parse  ../../{abfsPath}/{commitID}/b/; \
    #             ./joern-export --repr cpg14 --out ../../{abfsPath}/{commitID}/cpgsB/')

    return 0
