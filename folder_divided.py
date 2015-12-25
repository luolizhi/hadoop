# !first test
# _*_ coding: utf-8 _*_
import os
import shutil

old_path = "C:\Users\lukey\Desktop\jar\NBCorpus\Country"
new_path = "C:\Users\lukey\Desktop\jar\NBCorpus\\test"

for (dirpath,dirnames,filenames) in os.walk(old_path):
    for dirname in dirnames:#文件夹名
        dir_path = os.path.join(old_path,dirname)#获取文件夹的路径
        testpath = os.path.join(new_path,dirname)#新路径的文件夹名和之前的一样
        os.mkdir(testpath)#创建文件夹
        for (dpath,dnames,fnames) in os.walk(dir_path):#遍历每个文件夹下面的文件
            print dpath
            print len(fnames)
            for name in fnames[1::2]:#从第二个文件开始，每隔一个取一个，使训练集的文件不比测试集少
                filepath = os.path.join(dpath,name)
                shutil.move(filepath, testpath)