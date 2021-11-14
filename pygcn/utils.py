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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

import warnings


warnings.filterwarnings("ignore")

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


def del_node(dataname):
    path_original_root = r"../data/{}-original/".format(dataname)
    edges = np.genfromtxt(path_original_root + "edges.txt", dtype=np.int32)
    def_node = pd.read_csv(path_original_root + "del_node.csv").iloc[:, 1]
    idx_list = []
    for n in def_node:
        idx_tmp = np.where(edges == n)[0]
        idx_list.extend(list(idx_tmp))
    # or edges[:, 1] in def_node
    edges = np.delete(edges, idx_list, axis=0)
    np.savetxt(path_original_root + "edges.txt", edges, fmt='%d')
    print("done!")


#
# # 用于生成节点的度作为节点的属性
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
    idx = np.array(idx_features_labels[:, 0], dtype=np .int32)
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


def load_data_original(feature_type, dataname="cora", test_split_rate=0.7):
    print('Loading {} dataset...'.format(dataname))

    path_original_root = r"../data/{}-original/".format(dataname)
    idx_features_labels = pd.read_csv("{}/{}".format(path_original_root, feature_type))
    features = sp.csr_matrix(idx_features_labels.iloc[:, 1:], dtype=np.float32)

    # build graph
    idx = np.array(idx_features_labels.iloc[:, 0], dtype=np .int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}edges.txt".format(path_original_root),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx_features_labels.shape[0], idx_features_labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 生成原始网络的train val test 数据集
    length_edge_path_origin = edges_unordered.shape[0]
    idx_pos_train = range(int(length_edge_path_origin * 0.1))
    idx_pos_val = range(int(length_edge_path_origin * 0.1), int(length_edge_path_origin * 0.2))
    idx_pos_test = range(int(length_edge_path_origin * 0.2), length_edge_path_origin)
    idx_pos_test_train = range(int(length_edge_path_origin * 0.2),
                                  int(length_edge_path_origin * 0.2) + int(len(idx_pos_test) * test_split_rate))
    idx_pos_test_test = range(int(length_edge_path_origin * 0.2) + int(len(idx_pos_test) * test_split_rate),
                                 length_edge_path_origin)

    G = nx.Graph(edges.tolist())
    neg_sets = sample_neg_sets(G, n_samples=length_edge_path_origin, set_size=2)
    neg_sets = np.array(neg_sets)
    idx_neg_train = range(int(length_edge_path_origin * 0.1))
    idx_neg_val = range(int(length_edge_path_origin * 0.1), int(length_edge_path_origin * 0.2))
    idx_neg_test = range(int(length_edge_path_origin * 0.2), int(length_edge_path_origin * 1))
    idx_neg_test_train = idx_pos_test_train
    idx_neg_test_test = idx_pos_test_test

    idx_pos_train = torch.LongTensor(idx_pos_train)
    idx_pos_val = torch.LongTensor(idx_pos_val)
    idx_pos_test = torch.LongTensor(idx_pos_test)
    idx_pos_test_train = torch.LongTensor(idx_pos_test_train)
    idx_pos_test_test = torch.LongTensor(idx_pos_test_test)

    idx_neg_train = torch.LongTensor(idx_neg_train)
    idx_neg_val = torch.LongTensor(idx_neg_val)
    idx_neg_test = torch.LongTensor(idx_neg_test)
    idx_neg_test_train = torch.LongTensor(idx_neg_test_train)
    idx_neg_test_test = torch.LongTensor(idx_neg_test_test)

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, edges, neg_sets, idx_pos_train, idx_pos_val, idx_pos_test, idx_pos_test_train, idx_pos_test_test, idx_neg_train, idx_neg_val, idx_neg_test, idx_neg_test_train, idx_neg_test_test


def load_hon_data(edge_path=r"..\data\traces-simulated-original\edges.txt",
                  content_path=r"..\data\traces-simulated-original\traces.content"):
    """Load citation network dataset (cora only for now)"""
    print('Loading  dataset...')

    idx_features = pd.read_csv(content_path, dtype=np.dtype(str))
    print(idx_features)
    # features = sp.csr_matrix(idx_features.iloc[:, 1:-1], dtype=np.float32)
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


# 构建一个neg图，并进行label
def sample_neg_graph(G, Path_hon_root, Path_original_root, feature_type, n_samples=1000):
    '''
    :param G:
    :param Path_hon_root:
    :param Path_original_root:
    :param n_samples:
    :return:返回对应于高阶结点中的id，其中，每个list是一个二维的数据，
    每行中的id代表与原始网络中对应的高阶网络中的节点的id
    '''
    neg_sets = []
    n_nodes = G.number_of_nodes()
    nodes = G.nodes()
    nodes = list(nodes)
    while len(neg_sets) < n_samples:
        candid_set = [int(random.random() * n_nodes) for _ in range(2)]
        node1, node2 = nodes[candid_set[0]], nodes[candid_set[1]]
        if not G.has_edge(node1, node2):
            neg_sets.append([node1, node2])
    neg_sets = np.array(neg_sets)

    neg_sets = np.unique(neg_sets, axis=0)
    neg_sets = neg_sets.tolist()


    neg_sets = np.array(neg_sets)
    feature = pd.read_csv(Path_hon_root + feature_type)
    feature = feature.values.tolist()
    id_hon_map = []
    for l in feature:
        node1 = l[0].split("|")[0]
        id_hon_map.append(int(node1))

    id_hon_map = np.array(id_hon_map)
    node1_id_list = []
    node2_id_list = []
    c = set()
    for i in range(len(neg_sets)):
        n1, n2 = neg_sets[i, 0], neg_sets[i, 1]
        n1, n2 = int(n1), int(n2)
        n1_id = np.where(id_hon_map == n1)[0].tolist()
        n2_id = np.where(id_hon_map == n2)[0].tolist()
        if len(n1_id)!=0 and len(n2_id)!=0:
            node1_id_list.append(n1_id)
            node2_id_list.append(n2_id)
        else:
            print("映射有问题!")
            exit(0)
        # else:
        #     neg_sets[i, 0], neg_sets[i, 1] = tmp_n1, tmp_n2
        #     node1_id_list.append(node1_id_list[-1])
        #     node2_id_list.append(node1_id_list[-1])
        # if len(n1_id)==0:
        #     print("n1----", n1)
        #     c.add(n1)
        #     # exit(0)
        # if len(n2_id)==0:
        #     print("n2----", n2)
        #     c.add(n2)


    # tmp = pd.DataFrame(c)
    # tmp.to_csv('../data/click-stream-10-original/del_node.csv')
    # del_node('click-stream-10')
    f = open(Path_original_root + "edges_neg.txt", '+w')
    for i in neg_sets:
        s = [str(ss) for ss in i]
        s = ' '.join(s) + '\n'
        f.write(s)
    f.close()
    print("生成 neg graph done!")
    print("获取高阶结点中对应的id done！")
    node1_id_list = np.array(node1_id_list)
    node2_id_list = np.array(node2_id_list)
    return node1_id_list, node2_id_list


# edge_path_hon=r"..\data\traces-simulated\edges_label.txt",
# edge_path_origin="../data/traces-simulated-original/edges_label.txt",
# content_path_hon=r"..\data\traces-simulated\traces.content",
def load_hon_data_label(test_split_rate, dataname, feature_type):

    """Load citation network dataset (cora only for now)"""
    print('Loading  dataset...')
    path_hon_root = r"../data/{}/".format(dataname)
    path_original_root = r"../data/{}-original/".format(dataname)
    edge_path_hon = path_hon_root + "edges_label.txt"
    edge_path_origin = path_original_root + "edges_label.txt"
    content_path_hon = path_hon_root + "{}".format(feature_type)

    # 生成原始网络的train val test 数据集
    edge_origin = pd.read_csv(edge_path_origin)
    length_edge_path_origin = edge_origin.shape[0]
    idx_origin_train = range(int(length_edge_path_origin * 0.1))
    idx_origin_val = range(int(length_edge_path_origin * 0.1), int(length_edge_path_origin * 0.2))
    idx_origin_test = range(int(length_edge_path_origin * 0.2), int(length_edge_path_origin * 1))
    idx_origin_test_train =range(int(length_edge_path_origin * 0.2), int(length_edge_path_origin * 0.2) + int(len(idx_origin_test) * test_split_rate))
    idx_origin_test_test =range(int(length_edge_path_origin * 0.2) + int(len(idx_origin_test) * test_split_rate), int(length_edge_path_origin * 0.2) + int(len(idx_origin_test)))


    origin_train_label = edge_origin.iloc[idx_origin_train]['label'].tolist()
    origin_val_label = edge_origin.iloc[idx_origin_val]['label'].tolist()
    origin_test_label = edge_origin.iloc[idx_origin_test]['label'].tolist()
    idx_origin_test_train_label = edge_origin.iloc[idx_origin_test_train]['label'].tolist()
    idx_origin_test_test_label = edge_origin.iloc[idx_origin_test_test]['label'].tolist()

    origin_edge_test_test = edge_origin.iloc[idx_origin_test_test]

    idx_features = pd.read_csv(content_path_hon, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features.iloc[:, 1:], dtype=np.float32)

    # idx_mapping
    idx = idx_features['idx']
    idx_map = {j: i for i, j in enumerate(idx)}

    # 读取高阶网络的边
    edge_hon = pd.read_csv(edge_path_hon)
    edges_unordered = np.array(edge_hon.iloc[:, 1:3])
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    np.savetxt(path_hon_root + "edges_mapping.txt", edges, delimiter='	', fmt='%d')
    edges_with_labed =np.concatenate((edges, np.array(edge_hon.iloc[:, 3]).reshape(edges.shape[0], -1)), axis=1)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx_features.shape[0], idx_features.shape[0]),
                        dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    G = nx.Graph(edges[:, :2].tolist())


    edges_with_labed = pd.DataFrame(edges_with_labed, columns=['node1', 'node2', 'label'])
    edges_hon, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_neg_train, idx_neg_val, idx_neg_test,\
    id_pos_edge_test_train_hon, id_pos_edge_test_test_hon, idx_neg_test_train, idx_neg_test_test = \
        sample_pos_neg_sets_label(G, edges_with_labed, origin_train_label, origin_val_label, origin_test_label,
        idx_origin_test_train_label, idx_origin_test_test_label)


    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_pos_train = torch.LongTensor(idx_pos_train)
    idx_pos_val = torch.LongTensor(idx_pos_val)
    idx_pos_test = torch.LongTensor(idx_pos_test)
    id_pos_edge_test_train_hon = torch.LongTensor(idx_pos_train)
    id_pos_edge_test_test_hon = torch.LongTensor(id_pos_edge_test_test_hon)

    idx_neg_train = torch.LongTensor(idx_neg_train)
    idx_neg_val = torch.LongTensor(idx_neg_val)
    idx_neg_test = torch.LongTensor(idx_neg_test)
    idx_neg_test_train = torch.LongTensor(idx_neg_test_train)
    idx_neg_test_test = torch.LongTensor(idx_neg_test_test)

    return adj, features, edges_hon, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_neg_train, idx_neg_val, idx_neg_test, \
           id_pos_edge_test_train_hon, id_pos_edge_test_test_hon, idx_neg_test_train, idx_neg_test_test, origin_edge_test_test, \



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


def _link_prediction(loop, node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list,
                     y_test_train, y_test_test, cal_edge_method='Hadamard'):
    x_train = np.zeros((len(node1_emb_train_list), node1_emb_train_list.shape[1]))
    x_test = np.zeros((len(node1_emb_test_list), node2_emb_test_list.shape[1]))
    for i, (node1_emb, node2_emb) in enumerate(zip(node1_emb_train_list, node2_emb_train_list)):
        if cal_edge_method == 'Hadamard':
            x_train[i] = node1_emb * node2_emb
        elif cal_edge_method == 'Average':
            x_train[i] == np.add(node1_emb, node2_emb) * 0.5
    for i, (node1_emb, node2_emb) in enumerate(zip(node1_emb_test_list, node2_emb_test_list)):
        if cal_edge_method == 'Hadamard':
            x_test[i] = node1_emb * node2_emb
        elif cal_edge_method == 'Average':
            x_test[i] == np.add(node1_emb, node2_emb) * 0.5
    y_train = y_test_train
    y_test = y_test_test
    clf = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=5000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)[:, -1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

    eval_dict = {'auc': metrics.auc(fpr, tpr),
                 'pr': metrics.average_precision_score(y_test, y_score),
                 'recall': recall_score(y_test, y_pred, average='macro'),
                 'precision': precision_score(y_test, y_pred, average='macro'),
                 'f1': metrics.f1_score(y_test, y_pred),
                 'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                 'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
    if loop % 10 == 0:
        print(eval_dict)
    return eval_dict


def link_prediction(node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list, y_test_train, y_test_test, cal_edge_method, loop=50):
    print("link_prediction by {}".format("LR"))

    eval_dict = eval_dict = {'auc': 0.0, 'pr': 0.0, 'recall': 0.0, 'precision': 0.0,'f1': 0.0, 'f1-micro': 0.0, 'f1-macro': 0.0}
    for i in range(loop):
        tmp_dict = _link_prediction(i, node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list,
                                    y_test_train, y_test_test, cal_edge_method)
        for key in tmp_dict.keys():
            eval_dict[key] += tmp_dict[key]
    for key in tmp_dict.keys():
        eval_dict[key] = round((1.0 * eval_dict[key]) / loop, 4)
    print('average performance')
    print(eval_dict)
    return eval_dict

def _link_prediction_label(loop, node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list,
                           y_test_train, y_test_test, edge_test_test_hon_label, edge_test_test_origin_label, cal_edge_method):
    ft = np.zeros((len(node1_emb_train_list) + len(node1_emb_test_list), node1_emb_train_list.shape[1]))
    node1_emb_list = np.concatenate((node1_emb_train_list, node1_emb_test_list))
    node2_emb_list = np.concatenate((node2_emb_train_list, node2_emb_test_list))
    for i, (node1_emb, node2_emb) in enumerate(zip(node1_emb_list, node2_emb_list)):
        if cal_edge_method == 'Hadamard':
            ft[i] = node1_emb * node2_emb
        elif cal_edge_method == 'Average':
            ft[i] == np.add(node1_emb, node2_emb) * 0.5
    train_size = len(node1_emb_train_list)

    x_train = ft[:train_size]
    y_train = y_test_train
    x_test = ft[train_size:]
    y_test = y_test_test

    clf = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=5000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    lenght_pos_test = len(edge_test_test_hon_label)
    y_pos_pred = y_pred[:lenght_pos_test]
    y_neg_pred = y_pred[lenght_pos_test:]

    y_score = clf.predict_proba(x_test)[:, -1]
    y_score = y_score.reshape(y_score.shape[0], -1)
    y_pos_score = y_score[:lenght_pos_test]
    y_neg_score = y_score[lenght_pos_test:]


    edge_test_test_hon_label = edge_test_test_hon_label.reshape(y_pos_score.shape[0], -1)

    y_score_label = np.concatenate((y_pos_score, edge_test_test_hon_label), axis=1)
    y_score_label = pd.DataFrame(y_score_label, columns=['score', 'label'])
    y_score_label_mean = y_score_label.groupby('label').score.agg(['mean'])
    y_score = np.concatenate((y_score_label_mean, y_neg_score), axis=0)

    y_pred_label = np.concatenate((y_pos_pred, edge_test_test_hon_label), axis=1)
    y_pred_label = pd.DataFrame(y_pred_label, columns=['score', 'label'])
    y_pred_label_mean = y_pred_label.groupby('label').score.agg(['mean'])
    y_pred = np.concatenate((y_pred_label_mean, y_neg_pred), axis=0)

    threshold = 0.8

    y_pred_calss = [int(item > threshold) for item in y_pred]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    y_pred_score = [int(item > threshold) for item in y_score]
    y_pre = y_pred_calss
    eval_dict = {'auc': metrics.auc(fpr, tpr),
                 'pr': metrics.average_precision_score(y_test, y_pre),
                 'recall': recall_score(y_test, y_pre, average='macro'),
                 'precision': precision_score(y_test, y_pre, average='macro'),
                 'f1': metrics.f1_score(y_test, y_pre),
                 'f1-micro': metrics.f1_score(y_test, y_pre, average='micro'),
                 'f1-macro': metrics.f1_score(y_test, y_pre, average='macro')}
    if loop % 10 == 0:
        print(eval_dict)
    return eval_dict


def link_prediction_label(node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list, y_test_train, y_test_test,
                          edge_test_test_hon_label, edge_test_test_origin_label, cal_edge_method="Hadamard", loop=30):
    print("link_prediction by {}".format('LR'))

    eval_dict = {'auc': 0.0, 'pr': 0.0, 'recall': 0.0, 'precision': 0.0,'f1': 0.0, 'f1-micro': 0.0, 'f1-macro': 0.0}
    for i in range(loop):
        tmp_dict = _link_prediction_label(i, node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list,
                                          y_test_train, y_test_test, edge_test_test_hon_label, edge_test_test_origin_label, cal_edge_method)
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
def sample_pos_neg_sets_label(G, edge, origin_train_label, origin_val_label, origin_test_label,
                              idx_origin_test_train_label, idx_origin_test_test_label):
    # 高阶网络的正采样边
    id_pos_edge_train_hon = edge.loc[edge['label'].isin(origin_train_label)].index.tolist()
    id_pos_edge_val_hon = edge.loc[edge['label'].isin(origin_val_label)].index.tolist()
    id_pos_edge_test_hon = edge.loc[edge['label'].isin(origin_test_label)].index.tolist()
    id_pos_edge_test_train_hon = edge.loc[edge['label'].isin(idx_origin_test_train_label)].index.tolist()
    # tmp = edge.loc[edge['label'].isin(idx_origin_test_test_label)]
    id_pos_edge_test_test_hon = edge.loc[edge['label'].isin(idx_origin_test_test_label)].index.tolist()
    id_pos_edge_test_test_hon = np.array(id_pos_edge_test_test_hon)
    idx_origin_test_test_label = np.array(idx_origin_test_test_label)
    # 这里面的pos_edges 是带有label的高阶网络的边
    # tmp2 = edge.loc[edge['label'].isin(origin_test_label)]
    # pos_edges = pd.concat([edge.loc[edge['label'].isin(origin_train_label)], edge.loc[edge['label'].isin(origin_val_label)],
    #                        edge.loc[edge['label'].isin(origin_test_label)]])
    lenght_hon_pos_edges = len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon) + len(id_pos_edge_test_hon)


    neg_edges = np.array(sample_neg_sets(G, lenght_hon_pos_edges, set_size=2), dtype=np.int32)
    length_neg_edges = neg_edges.shape[0]
    tmp = neg_edges[-(lenght_hon_pos_edges - length_neg_edges):, :]  # 把去重删除的neg边补齐
    neg_edges = pd.DataFrame(np.append(neg_edges, tmp, axis=0), columns=['node1', 'node2'])
    neg_edges['label'] = -1
    length_neg_edges = neg_edges.shape[0]

    # 暂时写死，后面需要根据数据进行调整
    idx_neg_train = range(len(id_pos_edge_train_hon))
    idx_neg_val = range(len(id_pos_edge_train_hon), len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon))
    idx_neg_test = range(len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon), length_neg_edges)
    idx_neg_test_train = range(len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon), int(len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon) + len(idx_neg_test) * 0.7))
    idx_neg_test_test = range(len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon) + int(len(idx_neg_test) * 0.7), len(id_pos_edge_train_hon) + len(id_pos_edge_val_hon) + int(len(idx_neg_test) * 0.7) + len(idx_origin_test_test_label))

    return np.array(edge), np.array(neg_edges), id_pos_edge_train_hon, id_pos_edge_val_hon, \
           id_pos_edge_test_hon, idx_neg_train, idx_neg_val, idx_neg_test, id_pos_edge_test_train_hon, \
           id_pos_edge_test_test_hon, idx_neg_test_train, idx_neg_test_test

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
    # neg_sets = np.unique(neg_sets, axis=0)
    neg_sets = neg_sets.tolist()

    return neg_sets


def sample_test_originl(edge_path, n_samples, train_rate):
    edges = np.genfromtxt(edge_path, delimiter='	')
    edges = edges.tolist()
    G = nx.Graph(edges)
    neg_sets = []
    n_nodes = G.number_of_nodes()
    while len(neg_sets) < n_samples:
        candid_set = [int(random.random() * n_nodes) for _ in range(2)]
        for node1, node2 in combinations(candid_set, 2):
            if not G.has_edge(node1, node2):
                neg_sets.append(candid_set)
                break
    neg_sets = np.array(neg_sets)
    return neg_sets[:len(neg_sets) * train_rate, :], neg_sets[len(neg_sets) * train_rate:, :]


def collect_tri_sets(G):
    tri_sets = set(
        frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if
        G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i
