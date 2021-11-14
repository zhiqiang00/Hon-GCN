import random
from itertools import combinations

import networkx as nx
import pandas as pd
import numpy as np
import torch

from pygcn.utils import sample_neg_graph


def get_Graph(Path):
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


def get_Graph2(Path, node_id_mapping):
    edges = []
    f = open(Path)
    for line in f.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    f.close()
    G = nx.Graph(edges)
    nx.info(G)
    print("generate graph done!")
    return G

def get_degrees(graph_path, save_path, isSave):
    G = get_Graph(graph_path)
    print("正在生成特征（度）......")
    degrees = nx.degree(G)
    features = pd.DataFrame(degrees)
    features.columns = ['idx', 'degree']
    if isSave:
        features.to_csv(save_path, index=False)
    return features


def get_mapping(Path):
    nodes = []
    with open(Path) as ef:
        for line in ef.readlines():
            nodes.extend(line.strip().split()[:2])
    nodes = sorted(list(set(nodes)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    return node_id_mapping



def hon_to_originan(hon_path, origin_path, save_path_origin, save_path):
    print("开始进行映射")
    # 利用原始网络生成字典，即在原始网络中将每个边给一个id，然后进行映射
    with open(origin_path) as f:
        edge_dict = {}
        edge_original = []
        lines = f.readlines()
        for id, l in enumerate(lines):
            # print(l)
            node1, node2 = l.split()
            edge_original.append([node1, node2, id])
            key = node1 + '-' + node2
            edge_dict[key] = id

        edge_original = pd.DataFrame(edge_original, columns=['node1', 'node2', 'label'])
        edge_original.to_csv(save_path_origin, index=True)


    with open(hon_path) as f:
        edge_list = []  # 用于存放高阶网络中高阶表示的节点实际是原始网络中的那个节点。
        lines = f.readlines()
        for i, l in enumerate(lines):
            node1, node2 = l.split()
            node1 = node1.split(sep="|")[0]
            node2 = node2.split(sep="|")[0]
            # print("l, node1, node2: \nid=", i, l, node1, node2)
            label = edge_dict[node1 + '-' + node2]
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


def get_neg_test_hon_emb(features, test_neg_n_samples, feature_type, dataname="traces-1000"):
    # # 得到测试数据集中的test部分的neg的高阶网络的id
    path_hon_root = r"../data/{}/".format(dataname)
    path_original_root = r"../data/{}-original/".format(dataname)
    # f = open(path_original_root + 'edges.txt', "r")
    edges = np.genfromtxt(path_original_root + 'edges.txt',
                          dtype=np.dtype(str))
    # lines = f.readlines()
    G = nx.Graph(edges.tolist())
    node1_id_list, node2_id_list = sample_neg_graph(G, path_hon_root, path_original_root, feature_type, n_samples=test_neg_n_samples)
    features = np.array(features)
    node1_emb_list = []
    node2_emb_list = []
    for i in node1_id_list:
        node1_id_same = np.array(i)
        # print("node1_id_same", node1_id_same)
        embedding = features[node1_id_same]
        embedding = np.mean(embedding, axis=0)
        node1_emb_list.append(embedding)
    for i in node2_id_list:
        node2_id_same = np.array(i)
        embedding = features[node2_id_same]
        embedding = np.mean(embedding, axis=0)
        node2_emb_list.append(embedding)
    return torch.LongTensor(np.array(node1_emb_list)), torch.LongTensor(np.array(node2_emb_list))


def deepwalk_emb_pro(dataname, model_type):

    embpath = r"../data/embeddingforother/{}-original-{}.embeddings".format(dataname, model_type)
    degreepath = r"../data/{}-original/traces.degree".format(dataname)
    data = np.genfromtxt(embpath, skip_header=1)
    data = data[np.argsort(data[:, 0])]
    data = pd.DataFrame(data)
    degree = pd.read_csv(degreepath)
    data = data.iloc[:, 1:]
    data = pd.concat([degree['idx'], data], axis=1)
    # data.to_csv(embpath, index=False)
    return data

def get_origin(degreepath="../data/traces-1000-original/traces.degree",
               savepath='../data/traces-1000-original/edges_mapping.txt',
               edgepath='../data/traces-1000-original/edges.txt'):
    degreepath = pd.read_csv(degreepath)
    idx = degreepath['idx']
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(edgepath, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    np.savetxt(savepath, edges, delimiter='	', fmt='%d')

def edges_proc(dataname=""):
    # dataname = "traces-simulated"
    # dataname = "traces-1000"
    # dataname = "traces-10000"
    # dataname = "traces-100"
    # dataname = "click-stream"
    print("正在处理{}数据".format(dataname))
    path_original_root = r"../data/{}-original/".format(dataname)
    path_hon_root = r"../data/{}/".format(dataname)
    # 生成度
    get_degrees(isSave=True, graph_path=path_original_root + "edges.txt", save_path=path_original_root + "traces.degree")
    get_degrees(isSave=True, graph_path=path_hon_root + "edges.txt", save_path=path_hon_root + "traces.degree")
    print(" 生成度 done!")
    #
    # # 将高阶网络映射回原始网络
    hon_to_originan(hon_path=path_hon_root + "edges.txt",
                    origin_path=path_original_root + "edges.txt",
                    save_path=path_hon_root + "edges_label.txt",
                    save_path_origin=path_original_root + "edges_label.txt")

def get_degrees2(G, node_id_mapping, flag, save_path):
    print("正在生成特征（度）......")
    degrees = nx.degree(G)
    new_degrees = []
    max_degree = 0
    for node in degrees:
        new_degrees.append((list(node_id_mapping.keys())[list(node_id_mapping.values()).index(node[0])], node[1]))
        if node[1] > max_degree:
            max_degree = node[1]
    if flag == 'degree':
        dim = 1
        features = pd.DataFrame(list(new_degrees))
    elif flag == 'degree-onehot':
        dim = 100 if max_degree > 100 else max_degree
        new_degrees_onehot = []
        for node in new_degrees:
            temp = [node[0]]
            temp.extend([0] * dim)
            if node[1] > 100:
                temp[100] = 1
            else:
                temp[node[1]] = 1
            new_degrees_onehot.append(tuple(temp))
        features = pd.DataFrame(list(new_degrees_onehot))
    else:
        print('Error flag!')
        return
#    node_num = len(node_id_mapping.keys())
#    for i in range(node_num):
#        temp = [list(node_id_mapping.keys())[i]]
#        temp.extend([0] * node_num)
#        temp[i+1] = 1
#        new_degrees.append(tuple(temp))
    features.columns = ['idx'] + ['degree'] * dim
    save_file = save_path + flag + '.csv'
    features.to_csv(save_file, index=False)

def save_orginal_emb(path_original_root):
    degree = pd.read_csv(path_original_root + 'traces.degree')
    idx = degree['idx']
    idx_map = {j: i for i, j in enumerate(idx)}

    # 读取高阶网络的边
    edge = np.genfromtxt(path_original_root + 'edges.txt')
    edges_unordered = edge
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    np.savetxt(path_original_root + "edges_mapping.txt", edges, delimiter='	', fmt='%d')


def add_ho_edges():
    ho_edges = set()
    f = open("../data/click-stream-10/edges.txt", 'r')
    lines = f.readlines()
    for l in lines:
        node1, node2 = l.split()
        node1 = node1.split(sep="|")[0]
        node2 = node2.split(sep="|")[0]
        if (node1, node2) not in ho_edges:
            ho_edges.add((node1, node2))

    original_edges = []
    f = open("../data/click-stream-10-original/edges.txt", 'r')
    lines = f.readlines()
    for l in lines:
        n1, n2 = l.strip().split()
        original_edges.append((n1, n2))
    add_edges = []
    for oe in original_edges:
        if oe not in ho_edges:
            add_edges.append([oe[0] + '|', oe[1] + '|'])
    # ho_edges = np.genfromtxt("../data/click-stream-10/edges.txt")
    # ho_edges = np.concatenate((ho_edges, np.array(add_edges)), axis=0)
    f = open("../data/click-stream-10/test.txt", 'a+')
    for ae in add_edges:
        l = ae[0] + ' ' + ae[1]
        f.write(l+'\n')
    f.close()
    print('done!')





if __name__ == '__main__':
    # 生成度
    # dataname = "traces-simulated"
    # dataname = "traces-10000"
    dataname = "click-stream-10"
    # path_original_root = r"../data/{}-original/".format(dataname)
    path_hon_root = r"../data/{}/".format(dataname)


    # # 将微博数据抽取为序列
    # retwent_path = r"../data/weibo/Retweet_Content.txt"
    # num_path = r"../data/weibo/Weibo_Retweet_Num.txt"
    # save_path = r"../data/weibo/traces_weibo_threshold50-300.txt"
    # get_seq(retwent_path, num_path, save_path, thresholds=(50, 300))

    # 得到测试数据集中的test部分的neg的高阶网络的id
    # dataname = "traces-1000"
    # path_hon_root = r"../data/{}/".format(dataname)
    # path_original_root = r"../data/{}-original/".format(dataname)
    # # f = open(path_original_root + 'edges.txt', "r")
    # edges = np.genfromtxt(path_original_root + 'edges.txt',
    #                                     dtype=np.dtype(str))
    # # lines = f.readlines()
    # G = nx.Graph(edges.tolist())
    # node1_id_list, node2_id_list = sample_neg_graph(G, path_hon_root, path_original_root, n_samples=10000)
    # print(" get neg embedding")
    # 将利用deepwalk得到的embedding进一步处理
    # deepwalk_emb_pro()

    # # get_origin()
    # # edges_proc()
    # dataname = "click-stream-10"  # traces-10000  click-stream-10
    Path = '../data/{}/edges.txt'.format(dataname)
    node_id_mapping = get_mapping(Path)
    G = get_Graph2(Path, node_id_mapping)
    flag = 'degree-onehot'
    get_degrees2(G, node_id_mapping, flag, '../data/{}/'.format(dataname))
    # print("done!")
    # save_orginal_emb(path_original_root)
    # del_node("click-stream-10")
    # add_ho_edges()
    pass




