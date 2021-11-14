from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

from pygcn.data_processed import deepwalk_emb_pro
from pygcn.utils import link_prediction, load_data_original
import pandas as pd

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_neg_n_samples', type=int, default=1000,
                    help='test中负采样边的数量.')
parser.add_argument('--dataname', default="click-stream",  # traces-1000-original traces-1000, traces-simulated click-stream
                    help='Validate during training pass.')
parser.add_argument('--feature_type', default="degree-onehot.csv",  # traces.degree degree-onehot.csv
                    help='不同类型的节点特征.')
parser.add_argument('--model', default='GCN',
                    help='选择不用的模型进行训练')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, pos_edges, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_pos_test_train, idx_pos_test_test,\
#     idx_neg_train, idx_neg_val, idx_neg_test, idx_neg_test_train, idx_neg_test_test\
#         = load_data_original(dataname=args.dataname, feature_type=args.feature_type, test_split_rate=0.7)

def test(dataname, model_type):
    with torch.no_grad():
        # Load data
        adj, features, pos_edges, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_pos_test_train, idx_pos_test_test, \
        idx_neg_train, idx_neg_val, idx_neg_test, idx_neg_test_train, idx_neg_test_test \
            = load_data_original(dataname=dataname, feature_type=args.feature_type, test_split_rate=0.7)
        # output = model(features, adj)
        print("正在计算的是{}数据的{}模型".format(dataname, model_type))
        output = deepwalk_emb_pro(dataname, model_type=model_type)
        output = output.iloc[:, 1:].values
        # 获取test数据里面的作为train和test的节点list
        pos_node1_emb_train_list = pos_edges[idx_pos_test_train][:, 0]
        pos_node2_emb_train_list = pos_edges[idx_pos_test_train][:, 1]
        neg_node1_emb_train_list = neg_edges[idx_neg_test_train][:, 0]
        neg_node2_emb_train_list = neg_edges[idx_neg_test_train][:, 1]

        pos_node1_emb_test_list = pos_edges[idx_pos_test_test][:, 0]
        pos_node2_emb_test_list = pos_edges[idx_pos_test_test][:, 1]
        neg_node1_emb_test_list = neg_edges[idx_neg_test_test][:, 0]
        neg_node2_emb_test_list = neg_edges[idx_neg_test_test][:, 1]



        # 根据上面的id取出embedding
        pos_node1_train_emb = output[pos_node1_emb_train_list]
        pos_node2_train_emb = output[pos_node2_emb_train_list]
        neg_node1_train_emb = output[neg_node1_emb_train_list]
        neg_node2_train_emb = output[neg_node2_emb_train_list]

        pos_node1_test_emb = output[pos_node1_emb_test_list]
        pos_node2_test_emb = output[pos_node2_emb_test_list]
        neg_node1_test_emb = output[neg_node1_emb_test_list]
        neg_node2_test_emb = output[neg_node2_emb_test_list]

        node1_emb_train_list = np.append(pos_node1_train_emb, neg_node1_train_emb, axis=0)
        node2_emb_train_list = np.append(pos_node2_train_emb, neg_node2_train_emb, axis=0)

        node1_emb_test_list = np.append(pos_node1_test_emb, neg_node1_test_emb, axis=0)
        node2_emb_test_list = np.append(pos_node2_test_emb, neg_node2_test_emb, axis=0)

        y_test_train = np.array([1] * len(pos_node1_train_emb) + [0] * len(neg_node1_train_emb))
        y_test_test = np.array([1] * len(pos_node1_test_emb) + [0] * len(neg_node1_test_emb))

        # 运行链接预测代码
        link_prediction(node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list,
                              y_test_train, y_test_test, cal_edge_method="Hadamard", loop=10)
        print("{}数据的{}模型计算完成".format(dataname, model_type))


# Testing
# test('traces-10000', 'node2vec')
# test('traces-10000', 'LINE')
# test('traces-10000', 'SDNE')
# test('traces-10000', 'GF')

test("click-stream-10", 'node2vec')
test("click-stream-10", 'deepwalk')
