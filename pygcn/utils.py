import random
from itertools import combinations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_mapping(Path):
    nodes = []
    with open(Path) as ef:
        for line in ef.readlines():
            nodes.extend(line.strip().split()[:2])
    nodes = sorted(list(set(nodes)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    return node_id_mapping


def get_Graph(Path, node_id_mapping):
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



# 用于生成节点的度作为节点的属性
def get_degrees(G, path=r"D:\zhiqiang\coding\pygcn-master\data\traces-simulated-original\traces.content"):
    print("正在生成特征（度）......")
    degrees = nx.degree(G)
    features = pd.DataFrame(degrees)
    features.columns = ['idx', 'degree']
    features.to_csv(path, index=False)
    # print([G.degree[i] for i in range(num_nodes)])
    # return np.array([G.degree[i] for i in range(num_nodes)])


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    print(idx_map)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_hon_data(edge_path=r"..\data\traces-simulated-original\edges.txt",
                  content_path=r"..\data\traces-simulated-original\traces.content"):
    """Load citation network dataset (cora only for now)"""
    print('Loading  dataset...')

    idx_features = pd.read_csv(content_path, dtype=np.dtype(str))
    print(idx_features)
    features = sp.csr_matrix(idx_features.iloc[:, 1:-1], dtype=np.float32)
    features = sp.csr_matrix(np.expand_dims(idx_features.iloc[:, 1], 1), dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
#    idx = np.array([i for i in range(len(idx_features))], dtype=np.int32)  # 自己的高阶数据
    idx = idx_features.values[:, 0]
    idx_map = {j: i for i, j in enumerate(idx)}
    print(idx_map)
#    edges_unordered = np.genfromtxt(edge_path, dtype=np.int32)
    edges_unordered = []
    with open(edge_path) as file:
        line = file.readline()
        while line:
            line = line.split()
            edges_unordered.append([line[0], line[1]])
            line = file.readline()
    edges_unordered = np.array(edges_unordered)
#    print(list(map(idx_map.get, edges_unordered.flatten())))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), 
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx_features.shape[0], idx_features.shape[0]),
                        dtype=np.float32)

    # 跑GCN croa数据时使用

    # idx_features_labels = np.genfromtxt("../data/cora/cora.content", dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("../data/cora/cora.cites",
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    G = get_Graph(edge_path, idx_map)

    pos_edges, neg_edges, idx_pos_train, idx_pos_val, \
    idx_pos_test, idx_neg_train, idx_neg_val, idx_neg_test = sample_pos_neg_sets(G)

    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_pos_train = torch.LongTensor(idx_pos_train)
    idx_pos_val = torch.LongTensor(idx_pos_val)
    idx_pos_test = torch.LongTensor(idx_pos_test)

    idx_neg_train = torch.LongTensor(idx_neg_train)
    idx_neg_val = torch.LongTensor(idx_neg_val)
    idx_neg_test = torch.LongTensor(idx_neg_test)

    return adj, features, pos_edges, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_neg_train, idx_neg_val, idx_neg_test


def load_hon_data_label(edge_path_hon=r"..\data\traces-simulated\edges_label.txt",
                        edge_path_origin="../data/traces-simulated-original/edges_label.txt",
                        content_path=r"..\data\traces-simulated-original\traces.content"):
    """Load citation network dataset (cora only for now)"""
    print('Loading  dataset...')

    # 生成原始网络的train val test 数据集
    edge_origin = pd.read_csv(edge_path_origin)
    length_edge_path_origin = edge_origin.shape[0]
    idx_origin_train = range(int(length_edge_path_origin * 0.1))
    idx_origin_val = range(int(length_edge_path_origin * 0.1), int(length_edge_path_origin * 0.2))
    idx_origin_test = range(int(length_edge_path_origin * 0.2), int(length_edge_path_origin * 1))

    origin_train_label = edge_origin.iloc[idx_origin_train]['label']
    origin_val_label = edge_origin.iloc[idx_origin_val]['label']
    origin_test_label = edge_origin.iloc[idx_origin_test]['label']


    idx_features = pd.read_csv(content_path, dtype=np.dtype(str))
    features = sp.csr_matrix(np.expand_dims(idx_features.iloc[:, 1], 1), dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
#    idx = np.array([i for i in range(len(idx_features))], dtype=np.int32)  # 自己的高阶数据
    idx = idx_features.values[:, 0]
    idx_map = {j: i for i, j in enumerate(idx)}

    # 读取高阶网络的边
    edge_hon = pd.read_csv(edge_path_hon)
    edges_unordered = np.array(edge_hon.iloc[:, 1:3])
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx_features.shape[0], idx_features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # G = get_Graph(edges_unordered, idx_map)

    G = nx.Graph(edges)
    pos_edges, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_neg_train, idx_neg_val, idx_neg_test \
        = sample_pos_neg_sets_label(G, edge_hon, origin_train_label, origin_val_label, origin_test_label)

    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_pos_train = torch.LongTensor(idx_pos_train)
    idx_pos_val = torch.LongTensor(idx_pos_val)
    idx_pos_test = torch.LongTensor(idx_pos_test)

    idx_neg_train = torch.LongTensor(idx_neg_train)
    idx_neg_val = torch.LongTensor(idx_neg_val)
    idx_neg_test = torch.LongTensor(idx_neg_test)

    return adj, features, pos_edges, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_neg_train, idx_neg_val, idx_neg_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def _link_prediction(loop, node1_emb_list, node2_emb_list, labels, split_ratio=0.7, cal_edge_method='Hadamard', fit_method='LogisticRegression'):
    ft = np.zeros((len(node1_emb_list), node1_emb_list.shape[1]))
    for i, (node1_emb, node2_emb) in enumerate(zip(node1_emb_list, node2_emb_list)):
        if cal_edge_method == 'Hadamard':
            ft[i] = node1_emb * node2_emb
        elif cal_edge_method == 'Average':
            ft[i] == np.add(node1_emb, node2_emb) * 0.5
    labels = np.expand_dims(labels, axis=1)  # 把标签拓展一维，便于与embedding进行拼接
    ft = np.append(ft, labels, axis=1)
    np.random.shuffle(ft)
    train_size = int(len(node1_emb_list) * split_ratio)

    x_train = ft[:train_size, :-1]
    y_train = ft[:train_size, -1]
    x_test = ft[train_size:, :-1]
    y_test = ft[train_size:, -1]
    if fit_method == 'RandomForest':
        scaler = StandardScaler()  # 标准化转换
        scaler.fit(x_train)  # 训练标准化对象
        x_train = scaler.transform(x_train)  # 转换数据集

        # clf = RandomForestClassifier(criterion='entropy')
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print("==============================")
        print("accuracy_score", accuracy_score(y_pred, y_test))
        conf_mat = confusion_matrix(y_test, y_pred)
        print(conf_mat)
        print(classification_report(y_test, y_pred))
        return


    if fit_method == 'LogisticRegression':
        clf = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=5000)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_score = clf.predict_proba(x_test)[:, -1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

    eval_dict = {'auc': metrics.auc(fpr, tpr),
                 'pr': metrics.average_precision_score(y_test, y_score),
                 'f1': metrics.f1_score(y_test, y_pred),
                 'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                 'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
    if loop % 10 == 0:
        print(eval_dict)
    return eval_dict


def link_prediction(node1_emb_list, node2_emb_list, labels, cal_edge_method="Hadamard", split_ratio=0.7,
                    fit_method='LogisticRegression', loop=50):
    print("link_prediction by {}".format(fit_method))
    if fit_method == 'RandomForest':
        _link_prediction(node1_emb_list, node2_emb_list, labels, split_ratio, cal_edge_method, fit_method, i=0)
        return

    eval_dict = {'auc': 0.0, 'pr': 0.0, 'f1': 0.0, 'f1-micro': 0.0, 'f1-macro': 0.0}
    for i in range(loop):
        tmp_dict = _link_prediction(i, node1_emb_list, node2_emb_list, labels, split_ratio, cal_edge_method, fit_method)
        for key in tmp_dict.keys():
            eval_dict[key] += tmp_dict[key]
    for key in tmp_dict.keys():
        eval_dict[key] = round((1.0 * eval_dict[key]) / loop, 4)
    print('average performance')
    print(eval_dict)
    return eval_dict


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 采样正负节点
def sample_pos_neg_sets(G, data_usage=1.0):
    pos_edges = np.array(list(G.edges), dtype=np.int32)
    set_size = 2
    if data_usage < 1 - 1e-6:
        pos_edges, sample_i = retain_partial(pos_edges, ratio=data_usage)
    neg_edges = np.array(sample_neg_sets(G, pos_edges.shape[0], set_size=set_size), dtype=np.int32)
    length_pos_edges = pos_edges.shape[0]
    length_neg_edges = neg_edges.shape[0]
    tmp = neg_edges[-(length_pos_edges - length_neg_edges):, :]  # 把去重删除的neg边补齐
    # neg_edges = np.concatenate((neg_edges, tmp), axis=0)
    neg_edges = np.append(neg_edges, tmp, axis=0)
    length_neg_edges = neg_edges.shape[0]

    # 暂时写死，后面需要根据数据进行调整
    idx_pos_train = range(int(length_pos_edges * 0.1))
    idx_pos_val = range(int(length_pos_edges * 0.1), int(length_pos_edges * 0.2))
    idx_pos_test = range(int(length_pos_edges * 0.2), int(length_pos_edges * 1))

    idx_neg_train = range(int(length_neg_edges * 0.1))
    idx_neg_val = range(int(length_neg_edges * 0.1), int(length_neg_edges * 0.2))
    idx_neg_test = range(int(length_neg_edges * 0.2), int(length_neg_edges * 1))

    return np.array(pos_edges), np.array(
        neg_edges), idx_pos_train, idx_pos_val, idx_pos_test, idx_neg_train, idx_neg_val, idx_neg_test


# 采样正负节点 使用节点映射
def sample_pos_neg_sets_label(G, edge, origin_train_label, origin_val_label, origin_test_label, data_usage=1.0):
    # 高阶网络的正采样边
    id_pos_edge_train_hon = edge[edge['label'] in origin_train_label].index.tolist()
    id_pos_edge_val_hon = edge[edge['label'] in origin_val_label].index.tolist()
    id_pos_edge_test_hon = edge[edge['label'] in origin_test_label].index.tolist()
    pos_edges = pd.concat([edge[edge['label'] in origin_train_label], edge[edge['label'] in origin_val_label],
                           edge[edge['label'] in origin_test_label]])
    lenght_hon_pos_edges = len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon) + len(id_pos_edge_test_hon)


    set_size = 2
    neg_edges = np.array(sample_neg_sets(G, lenght_hon_pos_edges, set_size=set_size), dtype=np.int32)
    length_neg_edges = neg_edges.shape[0]
    tmp = neg_edges[-(lenght_hon_pos_edges - length_neg_edges):, :]  # 把去重删除的neg边补齐
    neg_edges = np.append(neg_edges, tmp, axis=0)
    length_neg_edges = neg_edges.shape[0]

    # 暂时写死，后面需要根据数据进行调整
    idx_neg_train = range(int(length_neg_edges * 0.1))
    idx_neg_val = range(int(length_neg_edges * 0.1), int(length_neg_edges * 0.2))
    idx_neg_test = range(int(length_neg_edges * 0.2), int(length_neg_edges * 1))

    return np.array(pos_edges), np.array(
        neg_edges), id_pos_edge_train_hon, id_pos_edge_val_hon, id_pos_edge_test_hon, idx_neg_train, idx_neg_val, idx_neg_test

def sample_neg_sets(G, n_samples, set_size):
    neg_sets = []
    n_nodes = G.number_of_nodes()
    max_iter = 1e9
    count = 0
    while len(neg_sets) < n_samples:
        count += 1
        if count > max_iter:
            raise Exception('Reach max sampling number of {}, input graph density too high'.format(max_iter))
        candid_set = [int(random.random() * n_nodes) for _ in range(set_size)]
        for node1, node2 in combinations(candid_set, 2):
            if not G.has_edge(node1, node2):
                neg_sets.append(candid_set)
                break
    neg_sets = np.array(neg_sets)
    neg_sets = np.unique(neg_sets, axis=0)
    neg_sets = neg_sets.tolist()

    return neg_sets


def collect_tri_sets(G):
    tri_sets = set(
        frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if
        G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i
