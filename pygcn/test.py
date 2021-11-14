def test_deepwalk():
    # idx_origin_test_train_neg, idx_origin_test_test_neg = sample_test_originl(sample_n=len(idx_origin_test),
    #                                                                           train_rate=test_split_rate)
    dataname = "traces-1000-original"
    deepwalk_emb_pro(dataname=dataname)
    idx_features = pd.read_csv(r'../data/{}/traces.content')
    output = idx_features.iloc[:, 1:]
    output = np.array(output)

    path_hon_root = r"../data/{}/".format(dataname)
    path_original_root = r"../data/{}-original/".format(dataname)
    edge_path_hon = path_hon_root + "edges_label.txt"
    edge_path_origin = path_original_root + "edges_label.txt"
    content_path_hon = path_hon_root + "traces.content"
    degree_path_hon = path_hon_root + "traces.degree"
    # 生成原始网络的train val test 数据集
    edge_origin = pd.read_csv(edge_path_origin)
    length_edge_path_origin = edge_origin.shape[0]
    idx_origin_train = range(int(length_edge_path_origin * 0.1))
    idx_origin_val = range(int(length_edge_path_origin * 0.1), int(length_edge_path_origin * 0.2))
    idx_origin_test = range(int(length_edge_path_origin * 0.2), int(length_edge_path_origin * 1))
    idx_origin_test_train = range(int(length_edge_path_origin * 0.2),
                                  int(length_edge_path_origin * 0.2) + int(len(idx_origin_test) * test_split_rate))
    idx_origin_test_test = range(int(length_edge_path_origin * 0.2) + int(len(idx_origin_test) * test_split_rate),
                                 int(length_edge_path_origin * 0.2) + int(len(idx_origin_test)))

    idx_origin_test_train, idx_origin_test_test, idx_origin_test_train_neg,  idx_origin_test_test_neg


    # 获取test数据里面的作为train和test的节点list
    pos_node1_emb_train_list = idx_origin_test_train[:, 0]
    pos_node2_emb_train_list = idx_origin_test_train[:, 1]
    neg_node1_emb_train_list = idx_origin_test_train_neg[:, 0]
    neg_node2_emb_train_list = idx_origin_test_train_neg[:, 1]

    pos_node1_emb_test_list = idx_origin_test_test[:, 0]
    pos_node2_emb_test_list = idx_origin_test_test[:, 1]
    neg_node1_emb_test_list = idx_origin_test_test_neg[:, 0]
    neg_node2_emb_test_list = idx_origin_test_test_neg[:, 1]

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
                    y_test_train, y_test_test, cal_edge_method="Hadamard", split_ratio=0.7, loop=50)