import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 取特征feature
    labels = encode_onehot(idx_features_labels[:, -1])  # one-hot label

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 节点
    idx_map = {j: i for i, j in enumerate(idx)}   # 构建节点的索引字典
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),  # 导入edge的数据
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)    # 将之前的转换成字典编号后的边
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # 构建边的邻接矩阵
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix，计算转置矩阵。将有向图转成无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)   # 对特征做了归一化的操作
    adj = normalize(adj + sp.eye(adj.shape[0]))   # 对A+I归一化
    # 训练，验证，测试的样本
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    # 将numpy的数据转换成torch格式
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
