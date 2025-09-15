import math
import os
import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from numpy import uint8, float32
from scipy.sparse import coo_matrix, lil_matrix

from IM.algo import line_graph


def load_data(data_path, dataset):
    """
    loads dataset

    assumes the following files to be present in the dataset directory:
    n: number of hypernodes
    """
    with open(os.path.join(data_path, f'{dataset}.txt'), 'rb') as handle:
        hyperedges = []
        lines = handle.readlines()
        max_index = 0
        dataset_special = ['iAF1260b', 'iJO1366', 'NDC-classes-full_graph']
        for line in lines:
            line = line.decode('utf-8')
            line = line.strip().split(" ")
            if dataset in dataset_special or 'hyperCL' in dataset:
                node_in_edge = [int(i) for i in line]
            else:
                node_in_edge = [int(i) - 1 for i in line]
            max_index = max(max(node_in_edge), max_index)
            hyperedges.append(node_in_edge)
        # print("number of hyperedges is", len(hypergraph))
    return hyperedges, max_index + 1


def load_data_sim(data_path, dataset):
    """
    loads dataset

    assumes the following files to be present in the dataset directory:
    n: number of hypernodes
    """
    with open(os.path.join(data_path, f'{dataset}-simplices.txt'), 'rb') as f1:
        nodes = [int(line.strip())-1 for line in f1 if line.strip()]
    with open(os.path.join(data_path, f'{dataset}-nverts.txt'), 'rb') as f2:
        hyperedges = []
        max_index = 0
        node_ptr = 0
        for line in f2:
            line = line.decode('utf-8')
            if not line.strip():
                continue
            n = int(line.strip())
            node_in_edge = nodes[node_ptr: node_ptr+n]
            max_index = max(max(node_in_edge), max_index)
            node_ptr += n
            if n == 1:
                continue
            hyperedges.append(node_in_edge)
        print("number of hyperedges is", len(hyperedges))
        print("average node number of hyperedges is ", np.mean([len(hyperedges[edge]) for edge in range(len(hyperedges))]))
    return hyperedges, max_index + 1


def generate_H(hyperedges, num_node):
    node_inc_edge = {n: set() for n in range(num_node)}
    num_edge = len(hyperedges)
    row = []
    column = []
    value = []
    for i, item in enumerate(hyperedges):
        for node in item:
            row.append(node)
            column.append(i)
            value.append(1)
            node_inc_edge[node].add(i)
    Incidence = coo_matrix((value, (row, column)), shape=(num_node, num_edge), dtype=uint8).tolil()
    return Incidence, node_inc_edge


def s_fun(x, threshold_low, threshold_high):
    if x <= threshold_low:
        return 0
    elif x >= threshold_high:
        return float(math.log(threshold_high + 1, math.e) / math.log(2, math.e))
    else:
        return float(math.log(x + 1, math.e) / math.log(2, math.e))


def threshold_init(number):
    """
    input: number of the nodes of the graph
    output: thresholds of all nodes, as a list
    """
    np.random.seed(1)
    return list(np.random.uniform(low=0, high=1, size=(number,)))


def draw_degree_dis(inc_matrix, dataset):
    edge_degree = np.sum(inc_matrix, axis=0).reshape(-1).tolist()[0]
    node_degree = np.sum(inc_matrix, axis=1).reshape(-1).tolist()[0]
    edge_degree_sort = set(sorted(edge_degree))
    node_degree_sort = set(sorted(node_degree))
    # edge_degree_fre = [edge_degree.count(k) for k in edge_degree_sort]
    # node_degree_fre = [node_degree.count(k) for k in node_degree_sort]
    # plt.loglog(list(edge_degree_sort), edge_degree_fre)
    # plt.title(f"edge degree distribution of {dataset}")
    # plt.show()
    # plt.loglog(list(node_degree_sort), node_degree_fre)
    # plt.title(f"node degree distribution of {dataset}")
    # plt.show()
    return node_degree_sort, edge_degree_sort


def hyperedge_weight_init(hyperedges):
    weights = []
    for edge in hyperedges:
        weight = {}
        for i, node in enumerate(edge):
            if i != len(edge) - 1:
                weight = 1 / len(edge)
                # weight[node] = random.uniform(0, 1 - sum(weight.values()))
            else:
                weight[node] = 1 - sum(weight.values())

        weights.append(weight)

    return weights


def edge_weight_for_nodes(H: np.matrix):
    node_num, edge_num = H.shape
    weights = np.zeros((node_num, edge_num))
    random.seed(1)
    for node in range(node_num):
        adj_edges = list(np.nonzero(H[node])[1])
        for i, edge in enumerate(adj_edges):
            if i == len(adj_edges) - 1:
                weight = 1 - sum(weights[node])
            else:
                weight = 1 / len(adj_edges)
            # weight = random.randint(0, sum(weights[node]))
            # weight = [1/len(adj_edges)]*len(adj_edges)
            weights[node, edge] = weight

    return weights


def node_edge_p(H):
    node_num, edge_num = H.shape
    p_node_to_edge = {i: {} for i in range(node_num)}
    p_edge_to_node = {i: {} for i in range(edge_num)}
    for node in range(node_num):
        adj_edges = list(np.nonzero(H[node])[1])
        for edge in adj_edges:
            p_node_to_edge[node][edge] = random.random()
            p_edge_to_node[edge][node] = random.random()

    return p_node_to_edge, p_edge_to_node


def compute_mth_component(A, m):
    A[A < m] = 0
    G = nx.from_numpy_array(A)
    connected_components = list(nx.connected_components(G))
    max_index, max_length = 0, 0
    for i, node_set in enumerate(connected_components):
        if len(node_set) > max_length:
            max_index, max_length = i, len(node_set)
    # print(m, connected_components[max_index])
    return connected_components[max_index]


def plot_HOC(inc_matrix, filename):
    num_node, num_edge = inc_matrix.shape
    A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    for i in range(num_node):
        A[i, i] = 0
    res_node = []
    for i in range(5):
        temp = compute_mth_component(A, i)
        # print(i, temp, len(temp))
        res_node.append(len(temp) / num_node)

    # plt.plot(range(len(res_node)), res_node)
    # plt.title(f"Node mth-connected component of {filename}")
    # plt.xlabel("m")
    # plt.ylabel(r"Size")
    # plt.xlim(0, None)
    # plt.ylim(None, 1)
    # k = 1 if len(res_node) < 10 else 5
    # x_major_locator = MultipleLocator(k)
    # ax = plt.gca()
    # # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.savefig(f'fig/connected/{filename}_Node.png')
    # plt.show()

    H_line = line_graph(inc_matrix).todense()
    res_edge = []
    for i in range(num_edge):
        H_line[i, i] = 0

    for i in range(5):
        temp = compute_mth_component(H_line, i)
        res_edge.append(len(temp) / num_edge)

    # plt.plot(range(len(res_edge)), res_edge)
    # plt.title(f"edge mth-connected component of {filename}")
    # plt.xlabel("m")
    # plt.ylabel(r"Size")
    # plt.xlim(0, None)
    # plt.ylim(None, 1)
    # k = 1 if len(res_edge) <= 10 else 5
    # x_major_locator = MultipleLocator(k)
    # ax = plt.gca()
    # # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.savefig(f'fig/connected/{filename}_edge.png')
    # plt.show()

    return res_node, res_edge


def compute_C(inc_matrix):
    N = inc_matrix.shape[0]
    diag_D_v = []
    for i in range(N):
        if sum(inc_matrix[i, :]) == 0:
            diag_D_v.append(1)
        else:
            diag_D_v.append(1 / sum(inc_matrix[i, :]))

    D_v = np.diag(diag_D_v)
    return inc_matrix @ inc_matrix.T @ D_v - np.identity(N)


from scipy.sparse import csr_matrix, diags, identity


def sparse_uint8_mult(A, B, chunk_size=1000):
    """分块乘法，强制 uint8 输出"""
    n_rows = A.shape[0]
    result = lil_matrix((n_rows, B.shape[1]), dtype=np.uint8)
    for i in range(0, n_rows, chunk_size):
        chunk = (A[i:i+chunk_size] @ B).astype(np.uint8)  # 立即降级
        result[i:i+chunk_size] = chunk
    return result.tocsr()


def compute_C_sparse(inc_matrix):
    # 确保输入是稀疏矩阵（CSR格式）
    if not isinstance(inc_matrix, csr_matrix):
        inc_matrix = csr_matrix(inc_matrix)

    N = inc_matrix.shape[0]

    # 计算 D_v 的对角元素
    row_sums = np.array(inc_matrix.sum(axis=1)).flatten()  # 计算每行的和
    diag_D_v = np.where(row_sums == 0, 1, 1 / row_sums)  # 处理零和的情况
    # 构建 D_v 稀疏对角矩阵
    D_v = diags(diag_D_v, dtype=float32)
    # 1. 计算 inc_matrix @ inc_matrix.T，并立即转为 uint8（确保值 ≤ 255）
    step1 = (inc_matrix @ inc_matrix.T).astype(np.uint8)  # 强制降级

    # 2. 计算 step1 @ D_v（uint8 @ float32 → 自动转为 float32）
    step2 = step1 @ D_v  # 结果是 float32

    # 3. 减去 identity_N（float32 - uint8 → 仍为 float32）
    C = step2 - identity(N, dtype=uint8)

    return C, step1

# H = np.array([[0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1]])
# print(compute_C(H))
