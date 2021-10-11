import networkx as nx
import pandas as pd
import numpy as np

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

def get_degrees(G, save_path=r"..\data\traces-simulated-original\traces.content"):
    print("正在生成特征（度）......")
    degrees = nx.degree(G)
    features = pd.DataFrame(degrees)
    features.columns = ['idx', 'degree']
    features.to_csv(save_path, index=False)


def hon_to_originan(hon_path, origin_path, save_path_origin, save_path):
    print("开始进行映射")
    # 利用原始网络生成字典，即在原始网络中将每个边给一个id，然后进行映射
    with open(origin_path) as f:
        edge_dict = {}
        edge_original = []
        lines = f.readlines()
        for id, l in enumerate(lines):
            node1, node2 = l.split()
            edge_original.append([node1, node2, id])
            key = node1 + '-' + node2
            edge_dict[key] = id
        edge_original = pd.DataFrame(edge_original, columns=['node1', 'node2', 'label'])
        edge_original.to_csv(save_path_origin, index=True)


    with open(hon_path) as f:
        edge_list = []  # 用于存放高阶网络中高阶表示的节点实际是原始网络中的那个节点。
        lines = f.readlines()
        for l in lines:
            node1, node2  = l.split()
            node1 = node1.split(sep="|")[0]
            node2 = node2.split(sep="|")[0]
            print("l, node1, node2", l, node1, node2)
            label = edge_dict[node1 + '-' + node2]
            edge_list.append([l.split()[0], l.split()[1], label])
        edge_list = pd.DataFrame(edge_list, columns=['node1', 'node2', 'label'])
        edge_list.to_csv(save_path, index=True)
        print("Done!")



if __name__ == '__main__':
    # 生成度
    # Path = r"../data/traces-simulated/edges.txt"
    # G = get_Graph(Path)
    # get_degrees(G, save_path=r"..\data\traces-simulated\traces.content")
    # print("done!")

    # 将高阶网络映射回原始网络
    hon_to_originan(hon_path="../data/traces-simulated/edges.txt",
                    origin_path="../data/traces-simulated-original/edges.txt",
                    save_path="../data/traces-simulated/edges_label.txt",
                    save_path_origin="../data/traces-simulated-original/edges_label.txt",)


