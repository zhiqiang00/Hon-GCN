import networkx as nx
import pandas as pd

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


if __name__ == '__main__':
    Path = r"../data/traces-simulated/edges.txt"
    G = get_Graph(Path)
    get_degrees(G, save_path=r"..\data\traces-simulated\traces.content")
    print("done!")