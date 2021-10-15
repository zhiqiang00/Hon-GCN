import random
from itertools import combinations

import networkx as nx
import pandas as pd
import numpy as np

from pygcn.utils import sample_neg_graph


def get_Graph(Path, node_id_mapping=0):
    edges = []
    f = open(Path)
    for line in f.readlines():
        node1, node2 = line.strip().split()[:2]
        # edges.append([node_id_mapping[node1], node_id_mapping[node2]])
        edges.append([node1, node2])
    f.close()
    G = nx.Graph(edges)
    nx.info(G)
    print("generate graph done!")
    return G


def get_degrees(G, save_path):
    print("正在生成特征（度）......")
    degrees = nx.degree(G)
    features = pd.DataFrame(degrees)
    features.columns = ['idx', 'degree']
    features.to_csv(save_path, index=False)


def hon_to_originan(hon_path, origin_path, save_path_origin, save_path, pos=1, isPos=True):
    print("开始进行映射")
    # 利用原始网络生成字典，即在原始网络中将每个边给一个id，然后进行映射
    with open(r"../data/traces-1000-original/edges_neg.txt") as  f:
        edge_dict = {}
        edge_original = []
        lines = f.readlines()
        for id, l in enumerate(lines):
            node1, node2 = l.split()
            edge_original.append([node1, node2, id * pos])
            key = node1 + '-' + node2
            edge_dict[key] = id * pos - 1

        edge_original = pd.DataFrame(edge_original, columns=['node1', 'node2', 'label'])
        edge_original.to_csv(save_path_origin, index=True)


    with open(hon_path) as f:
        edge_list = []  # 用于存放高阶网络中高阶表示的节点实际是原始网络中的那个节点。
        lines = f.readlines()
        if isPos:
            for i, l in enumerate(lines):
                node1, node2 = l.split()
                node1 = node1.split(sep="|")[0]
                node2 = node2.split(sep="|")[0]
                print("l, node1, node2: \nid=", i, l, node1, node2)
                label = edge_dict[node1 + '-' + node2]
                edge_list.append([l.split()[0], l.split()[1], label])
        else:
            for i, l in enumerate(lines):
                node1, node2 = l.split()
                node1 = node1.split(sep="|")[0]
                node2 = node2.split(sep="|")[0]
                key = node1 + '-' + node2
                print("l, node1, node2: \nid=", i, l, node1, node2)
                if edge_dict.get(key):
                    print(edge_dict.get(key, "JKKKKKKKK"))
                    label = edge_dict[node1 + '-' + node2]
                else:
                    label = 0
                edge_list.append([l.split()[0], l.split()[1], label])
        edge_list = pd.DataFrame(edge_list, columns=['node1', 'node2', 'label'])
        edge_list.to_csv(save_path, index=True)
        print("Done!")


def get_weibo_message(num_path, thresholds):
    thresholds_min, thresholds_max =  thresholds
    data = np.genfromtxt(num_path, delimiter='	', dtype=np.int64)
    data_thresholds = data[data[:, 1] > thresholds_min]
    data_thresholds = data_thresholds[data_thresholds[:, 1] < thresholds_max]
    # data = pd.DataFrame(data, columns=['id', 'num'])
    # tmp = data.groupby(['num']).agg(['count'])
    return data[:, 0], data_thresholds[:, 0]


# 将微博数据抽取为序列
def get_seq(retwent_path, num_path, save_path, thresholds):
    messageid, messageid_thresholds = get_weibo_message(num_path, thresholds)
    messageid = [str(i) for i in messageid]
    messageid = set(messageid)
    messageid_thresholds = [str(i) for i in messageid_thresholds]
    messageid_thresholds = set(messageid_thresholds)
    seq_retweet = []
    f = open(retwent_path, '+r')
    line = f.readline()
    tmp = []
    i = 0
    while line:
        l = line.strip().split()
        if l and l[0] in messageid:
            if tmp and tmp[0] in messageid_thresholds:
                seq_retweet.append(tmp)
                if i % 500 == 0:
                    print(i, '  '.join(tmp))
                i += 1
            tmp = []
            tmp.append(l[0])
            tmp.append(l[1])
        elif l and len(l[0]) == 10 and l[0] != tmp[-1]:
            tmp.append(l[0])
        line = f.readline()
    f.close()
    seq_retweet = seq_retweet[1:]
    # seq_retweet = [[int(j) for j in i] for i in seq_retweet]
    f = open(save_path, '+w')
    for s in seq_retweet:
        ss = " ".join(s) + '\n'
        f.write(ss)
    f.close()
    print("weibo 序列数据抽取完成！")



if __name__ == '__main__':
    # 生成度
    # dataname = "traces-simulated"
    # dataname = "traces-1000"
    # path_original_root = r"../data/{}-original/".format(dataname)
    # path_hon_root = r"../data/{}/".format(dataname)
    # G = get_Graph(path_original_root + "edges.txt")
    # get_degrees(G, save_path=path_original_root + "traces.content")
    #
    # G = get_Graph(path_hon_root + "edges.txt")
    # get_degrees(G, save_path=path_hon_root + "traces.content")
    # print(" 生成度 done!")
    #
    # # 将高阶网络映射回原始网络
    # hon_to_originan(hon_path=path_hon_root + "edges.txt",
    #                 origin_path=path_original_root + "edges.txt",
    #                 save_path=path_hon_root + "edges_label.txt",
    #                 save_path_origin=path_original_root + "edges_label.txt", pos=1)

    # # 将微博数据抽取为序列
    # retwent_path = r"../data/weibo/Retweet_Content.txt"
    # num_path = r"../data/weibo/Weibo_Retweet_Num.txt"
    # save_path = r"../data/weibo/traces_weibo_threshold50-300.txt"
    # get_seq(retwent_path, num_path, save_path, thresholds=(50, 300))

    # # 得到测试数据集中的test部分的neg的高阶网络的id
    dataname = "traces-1000"
    path_hon_root = r"../data/{}/".format(dataname)
    path_original_root = r"../data/{}-original/".format(dataname)
    # f = open(path_original_root + 'edges.txt', "r")
    edges = np.genfromtxt(path_original_root + 'edges.txt',
                                        dtype=np.dtype(str))
    # lines = f.readlines()
    G = nx.Graph(edges.tolist())
    sample_neg_graph(G, path_hon_root, path_original_root, n_samples=10000)



