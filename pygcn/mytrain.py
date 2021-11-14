from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from pygcn.data_processed import get_neg_test_hon_emb, deepwalk_emb_pro, edges_proc
from pygcn.utils import load_hon_data_label, link_prediction_label, link_prediction
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_neg_n_samples', type=int, default=10000,
                    help='test中负采样边的数量.')
parser.add_argument('--dataname', default="click-stream-10",  # traces-1000, traces-simulated, click-stream
                    help='Validate during training pass.')
parser.add_argument('--feature_type', default="degree-onehot.csv",  # traces.degree degree-onehot.csv
                    help='不同类型的节点特征.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# process data
edges_proc(dataname=args.dataname)
# Load data
adj, features, edges_hon, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_neg_train, idx_neg_val, \
idx_neg_test, id_pos_edge_test_train_hon, id_pos_edge_test_test_hon, idx_neg_test_train, idx_neg_test_test, \
origin_edge_test_test = load_hon_data_label(test_split_rate=0.7, dataname=args.dataname, feature_type=args.feature_type)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=7,  # 最后输出结果为7维
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 最后需要修改
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    edges_hon = edges_hon.cuda()
    neg_edges = neg_edges.cuda()
    idx_pos_train = idx_pos_train.cuda()
    idx_pos_val = idx_pos_val.cuda()
    idx_pos_test = idx_pos_test.cuda()
    idx_neg_train = idx_neg_train.cuda()
    idx_neg_val = idx_neg_val.cuda()
    idx_neg_test = idx_neg_test.cuda()



def train(epoch):
    pos_node1_emb_list = edges_hon[idx_pos_train][:, 0]
    pos_node2_emb_list = edges_hon[idx_pos_train][:, 1]
    neg_node1_emb_list = neg_edges[idx_neg_train][:, 0]
    neg_node2_emb_list = neg_edges[idx_neg_train][:, 1]

    # pos_node = np.unique(np.vstack((pos_node1_emb_list, pos_node2_emb_list)))
    # neg_node = np.unique(np.vstack((neg_node1_emb_list, neg_node2_emb_list)))
    # pos_features = features[pos_node, :]
    # # neg_features = features[neg_node, :]
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    pos_node1_emb = output[pos_node1_emb_list]
    pos_node2_emb = output[pos_node2_emb_list]
    neg_node1_emb = output[neg_node1_emb_list]
    neg_node2_emb = output[neg_node2_emb_list]


    pos_node1_emb = pos_node1_emb.view(-1, 1, pos_node1_emb.shape[1])
    pos_node2_emb = pos_node2_emb.view(-1, pos_node2_emb.shape[1], 1)
    neg_node1_emb = neg_node1_emb.view(-1, 1, neg_node1_emb.shape[1])
    neg_node2_emb = neg_node2_emb.view(-1, neg_node2_emb.shape[1], 1)

    pos_out = torch.bmm(pos_node1_emb, pos_node2_emb)
    neg_out = -torch.bmm(neg_node1_emb, neg_node2_emb)
    train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
    train_loss.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    # validation
    model.eval()
    val_loss = []
    with torch.no_grad():
        pos_node1_emb_list = edges_hon[idx_pos_val][:, 0]
        pos_node2_emb_list = edges_hon[idx_pos_val][:, 1]
        neg_node1_emb_list = neg_edges[idx_neg_val][:, 0]
        neg_node2_emb_list = neg_edges[idx_neg_val][:, 1]

        t = time.time()
        pos_node1_emb = output[pos_node1_emb_list]
        pos_node2_emb = output[pos_node2_emb_list]
        neg_node1_emb = output[neg_node1_emb_list]
        neg_node2_emb = output[neg_node2_emb_list]

        pos_node1_emb = pos_node1_emb.view(-1, 1, pos_node1_emb.shape[1])
        pos_node2_emb = pos_node2_emb.view(-1, pos_node2_emb.shape[1], 1)
        neg_node1_emb = neg_node1_emb.view(-1, 1, neg_node1_emb.shape[1])
        neg_node2_emb = neg_node2_emb.view(-1, neg_node2_emb.shape[1], 1)

        pos_out = torch.bmm(pos_node1_emb, pos_node2_emb)
        neg_out = -torch.bmm(neg_node1_emb, neg_node2_emb)
        val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
        val_loss = torch.mean(torch.tensor(val_loss))

    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(train_loss.item()),
              'loss_val: {:.4f}'.format(val_loss.item()),
              'time: {:.4f}s'.format(time.time() - t))



def test_hon():
    print("正在保存模型......")
    torch.save(model, "../model/{}_{}_ho_model.pt".format(args.dataname, args.feature_type))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        # 获取test数据里面的作为train和test的节点list
        pos_node1_emb_train_list = edges_hon[id_pos_edge_test_train_hon][:, 0]
        pos_node2_emb_train_list = edges_hon[id_pos_edge_test_train_hon][:, 1]
        neg_node1_emb_train_list = neg_edges[idx_neg_test_train][:, 0]
        neg_node2_emb_train_list = neg_edges[idx_neg_test_train][:, 1]

        pos_node1_emb_test_list = edges_hon[id_pos_edge_test_test_hon][:, 0]
        pos_node2_emb_test_list = edges_hon[id_pos_edge_test_test_hon][:, 1]
        # neg_node1_emb_test_list = neg_edges[idx_neg_test_test][:, 0]
        # neg_node2_emb_test_list = neg_edges[idx_neg_test_test][:, 1]



        # 根据上面的id取出embedding
        pos_node1_train_emb = output[pos_node1_emb_train_list]
        pos_node2_train_emb = output[pos_node2_emb_train_list]
        neg_node1_train_emb = output[neg_node1_emb_train_list]
        neg_node2_train_emb = output[neg_node2_emb_train_list]

        pos_node1_test_emb = output[pos_node1_emb_test_list]
        pos_node2_test_emb = output[pos_node2_emb_test_list]
        # neg_node1_test_emb = output[neg_node1_emb_test_list]
        # neg_node2_test_emb = output[neg_node2_emb_test_list]
        neg_node1_test_emb, neg_node2_test_emb = get_neg_test_hon_emb(output, dataname=args.dataname, feature_type=args.feature_type,
                                                                      test_neg_n_samples=args.test_neg_n_samples)

        node1_emb_train_list = np.append(pos_node1_train_emb, neg_node1_train_emb, axis=0)
        node2_emb_train_list = np.append(pos_node2_train_emb, neg_node2_train_emb, axis=0)

        node1_emb_test_list = np.append(pos_node1_test_emb, neg_node1_test_emb, axis=0)
        node2_emb_test_list = np.append(pos_node2_test_emb, neg_node2_test_emb, axis=0)
        # node1_emb_test_list = pos_node1_test_emb
        # node2_emb_test_list = pos_node2_test_emb

        edge_test_test_hon_label = edges_hon[id_pos_edge_test_test_hon, -1]
        edge_test_test_origin_label = np.array(origin_edge_test_test['label'])
        y_test_train = np.array([1] * len(id_pos_edge_test_train_hon) + [0] * len(idx_neg_test_train))
        y_test_test = np.array([1] * len(origin_edge_test_test) + [0] * neg_node1_test_emb.shape[0])


        # 运行链接预测代码
        link_prediction_label(node1_emb_train_list, node2_emb_train_list, node1_emb_test_list, node2_emb_test_list,
                              y_test_train, y_test_test, edge_test_test_hon_label, edge_test_test_origin_label, cal_edge_method="Hadamard", loop=10)

    #     pos_node1_emb = pos_node1_emb.view(-1, 1, pos_node1_emb.shape[1])
    #     pos_node2_emb = pos_node2_emb.view(-1, pos_node2_emb.shape[1], 1)
    #     neg_node1_emb = neg_node1_emb.view(-1, 1, neg_node1_emb.shape[1])
    #     neg_node2_emb = neg_node2_emb.view(-1, neg_node2_emb.shape[1], 1)
    #
    #     pos_out = torch.bmm(pos_node1_emb, pos_node2_emb).flatten()
    #     neg_out = -torch.bmm(neg_node1_emb, neg_node2_emb).flatten()
    #     pos_proba_list.append(torch.sigmoid(pos_out))
    #     neg_proba_list.append(torch.sigmoid(neg_out))
    #
    #     y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
    #     y_proba_test = y_proba_test.cpu().numpy()
    # auc = roc_auc_score(y_true_test, y_proba_test)
    # ap = average_precision_score(y_true_test, y_proba_test)
    # print('Link Prediction Test')
    # print('AUC = {}'.format(auc))
    # print('AP = {}'.format(ap))
    # auc_list.append(auc)
    # ap_list.append(ap)
    # print('----------------------------------------------------------------')
    # print('Link Prediction Tests Summary')
    # print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    # print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test_hon()

