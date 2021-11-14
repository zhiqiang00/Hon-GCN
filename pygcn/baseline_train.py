from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import link_prediction, load_data_original
from pygcn.models import GCN

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
parser.add_argument('--dataname', default="traces-10000",  # traces-1000-original traces-1000, traces-simulated click-stream
                    help='Validate during training pass.')
parser.add_argument('--feature_type', default="traces.degree",  # traces.degree degree-onehot.csv
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
adj, features, pos_edges, neg_edges, idx_pos_train, idx_pos_val, idx_pos_test, idx_pos_test_train, idx_pos_test_test,\
    idx_neg_train, idx_neg_val, idx_neg_test, idx_neg_test_train, idx_neg_test_test\
        = load_data_original(dataname=args.dataname, feature_type=args.feature_type, test_split_rate=0.7)

# Model and optimizer
if args.model == "Deepwalk":
    pass
elif args.model == "Node2Vec":
    pass
elif args.model == "GCN":
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=7,  # 最后输出结果为7维
                dropout=args.dropout)
elif args.model == "GAT":
    pass
elif args.model == "GraphSage":
    pass
elif args.model == "GemoGCN":
    pass
else:
    print("模型输入错误或不在预设范围内！！！")
    exit(0)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 最后需要修改
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()

    idx_pos_train = idx_pos_train.cuda()
    idx_pos_val = idx_pos_val.cuda()
    idx_pos_test = idx_pos_test.cuda()
    idx_pos_test_train = idx_pos_test_train.cuda()
    idx_pos_test_test = idx_pos_test_test.cuda()
    idx_neg_train = idx_neg_train.cuda()
    idx_neg_val = idx_neg_val.cuda()
    idx_neg_test = idx_neg_test.cuda()
    idx_neg_test_train = idx_neg_test_train.cuda()
    idx_neg_test_test = idx_neg_test_test.cuda()



def train(epoch):
    pos_node1_emb_list = pos_edges[idx_pos_train][:, 0]
    pos_node2_emb_list = pos_edges[idx_pos_train][:, 1]
    neg_node1_emb_list = neg_edges[idx_neg_train][:, 0]
    neg_node2_emb_list = neg_edges[idx_neg_train][:, 1]


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
        pos_node1_emb_list = pos_edges[idx_pos_val][:, 0]
        pos_node2_emb_list = pos_edges[idx_pos_val][:, 1]
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



def test():
    print("正在保存模型......")
    torch.save(model, "../model/{}_{}_model.pt".format(args.dataname, args.feature_type))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)

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



# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

