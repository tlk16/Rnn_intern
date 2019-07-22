import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split
import string
import torch
import re
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


data_dir = {'train': ('swc_data/swc_v1/train/interneuron', 'swc_data/swc_v1/train/principal cell/microglia', 'swc_data/swc_v1/train/principal cell/pyramidal', 'swc_data/swc_v1/train/Glia', 'swc_data/swc_v1/train/Not reported', 'swc_data/swc_v1/train/sensory receptor'),
            'test': ('swc_data/swc_v1/test/interneuron', 'swc_data/swc_v1/test/principal cell/microglia', 'swc_data/swc_v1/test/principal cell/pyramidal', 'swc_data/swc_v1/test/Glia', 'swc_data/swc_v1/test/Not reported', 'swc_data/swc_v1/test/sensory receptor')}

# (num_data, maxlen_of_tree, len_of_line )

# Dataset 类， 输入某个元组，元组中每个路径是包含一类的文件夹
class RNNdataset(Dataset):
    def __init__(self, phase, data_dir=data_dir, maxlen_of_tree=300, len_of_line=7):
        self.phase = phase
        self.data_dir = data_dir[phase]
        self.maxlen_of_tree = maxlen_of_tree
        self.file_list = self.list_file()
        self.len_of_line = len_of_line


    def __getitem__(self, item):
        # 重载函数，返回值：
        # datum np.array shape = (maxlen_of_tree, 10)
        # label = np.array shape = (1,)
        # tree_len int 样本的实际长度，即除去填充的0的部分

        c = 0
        while item > sum(len(fi) for fi in self.file_list[0:c+1]):
            c += 1

        filename = self.file_list[c][item - 1 - sum(len(fi) for fi in self.file_list[:c])]
        label = c

        lines = self.readswc(filename)
        datum = np.zeros([self.maxlen_of_tree, self.len_of_line])
        tree_len = len(lines)
        for i, item in enumerate(lines):
            datum[i] = np.array(item)

        return datum, label, tree_len


    def __len__(self):
        return sum([len(fi) for fi in self.file_list])

    def list_file(self):
        ls = [[] for d in self.data_dir]
        file_list = [[] for d in self.data_dir]

        for i in range(len(ls)):
            self.recur_listdir(self.data_dir[i], ls[i])

        # 返回swc文件列表，其中不包括树长度大于maxlen的文件

        print('reading data list ... ')
        # wash 掉长度太长的
        for i, d in enumerate(ls):
            for filename in tqdm(d):
                with open(filename, 'r') as f:
                    lines = f.read().strip().split('\n')

                j = 0
                while lines[j].startswith('#'):
                    j += 1
                lines = lines[j:]  # 去除注释行

                if len(lines) <= self.maxlen_of_tree:
                    file_list[i].append(filename)

        # 样本均衡， 将所有类的样本数扩充到一致
        if self.phase == 'train':
            m = max([len(fi) for fi in file_list])
            for fi in file_list:
                if len(fi) < m:
                    extra = np.random.choice(fi, size=m - len(fi))
                    fi.extend(extra)

        print('RNNdataset: ', self.phase, [len(fi) for fi in file_list])

        return file_list


    def recur_listdir(self, path, dir_list):
        for f in os.listdir(path):
            if os.path.isdir(path + '/' + f):
                self.recur_listdir(path + '/' + f, dir_list)
            else:
                dir_list.append(path + '/' + f)

    def readswc(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')

        i = 0
        while lines[i].startswith('#'):
            i += 1
        lines = lines[i:]  # 去除注释行

        for i in range(len(lines)):
            lines[i] = re.split(r'[\[\],\s]', lines[i])
            while '' in lines[i]:
                lines[i].remove('')

        return lines





if __name__ == '__main__':
    # 改用lstm,   没有明显效果
    # 归一化，
    # 去掉某些项，比如id   去掉id和坐标后几乎无区别 / 只保留坐标情况下 似乎训练慢一些， 区别不大 / 全部为1或random， 达不到60
    # 使用 recursive tree model 正在调研
    # 比较两棵树相似度的方法， 把树转化为字符串

    # 数据可视化
    # 考虑编程上的错误

    # 测试集提供一个reward：当发现测试错误率增大时，回退半步。

    from mpl_toolkits import mplot3d #有用
    import matplotlib.pyplot as plt
    import numpy as np
    len0 = []
    len1 = []
    c = RECTREEdataset()
    for file in c.file_list:
        filename = c.dataset_dir + '/' + file
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')
            lines = lines[4:]  # 去除注释行

        for i in range(len(lines)):
            lines[i] = re.split(r'[\[\],\s]', lines[i])
            while '' in lines[i]:
                lines[i].remove('')
        tree_len = len(lines)
        label = c.label_dicts[int(file.split('.')[0])]
        if label == 0:
            len0.append(tree_len)
        elif label == 1:
            len1.append(tree_len)
        else:
            print(label)
    plt.hist(x=[len0, len1], bins=50)
    plt.show()



    # 可视化 ###########################################################

    ax = plt.axes(projection='3d')

    filename = dataset_dir + '/' + '100.swc'
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
        lines = lines[4:]               #去除注释行

    for i in range(len(lines)):
        lines[i] = re.split(r'[\[\],\s]', lines[i])
        while '' in lines[i]:
            lines[i].remove('')
    print(lines)
    lines = np.array(lines, dtype='float32')
    print(lines)
    from sklearn.preprocessing import normalize

    # ax.plot3D(lines[:, 2], lines[:, 3], lines[:, 4], 'gray')
    ax.scatter3D(lines[:, 1], lines[:, 2], lines[:, 3])
    for line in lines:
        ax.plot3D([line[1], lines[int(line[5]), 1]], [line[2], lines[int(line[5]), 2]], [line[3], lines[int(line[5]), 3]])

    plt.show()
    ############################################################################
