"""
    split a data set with size("data_size") into train, validation and test-set

"""


import os
import shutil

data_size = 1200

path = "data\images"
trainset_path='data\\train'
valset_path='data\\val'
testset_path='data\\test'
for p in [trainset_path,valset_path,testset_path]:
    if not os.path.exists(p):
        os.makedirs(p)
    shutil.rmtree(p)
    os.mkdir(p)

files = os.listdir(path)
for i in range(data_size):
    file = files[i]

    fp = os.path.join(path, file)
    if i % 10 < 7:
        shutil.copyfile(fp, os.path.join(trainset_path, file))
    elif 7 <= i % 10 < 9:
        shutil.copyfile(fp, os.path.join(valset_path, file))
    else:
        shutil.copyfile(fp, os.path.join(testset_path, file))
