import copy
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from Hyperspreading import Hyperspreading
from IM.algo import seeds_choose_v3, seeds_choose_v0506, seeds_choose_v0619, seeds_choose_v0630
from IM.utils import compute_C

matplotlib.use('Agg')
plt.switch_backend('agg')
hs = Hyperspreading()


def getSeeds_sta(degree, i):
    # print(i)
    matrix = [np.arange(len(degree)), degree]
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
    degree_list = list(df_sort_matrix.loc['node_degree'])
    nodes_list = list(df_sort_matrix.loc['node_index'])
    chosen_arr = list(df_sort_matrix.loc['node_index'][:i])
    index = np.where(np.array(degree_list) == degree_list[i])[0]
    nodes_set = list(np.array(nodes_list)[index])
    while 1:
        node = random.sample(nodes_set, 1)[0]
        if node not in chosen_arr:
            chosen_arr.append(node)
            break
        else:
            nodes_set.remove(node)
            continue
    return chosen_arr


def DegreeMax(df_hyper_matrix, K, R, N, beta, T):
    """
    Degree algorithm
    """
    degree = getTotalAdj(df_hyper_matrix, N)
    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(0, K):
            if i == K - 1:
                time_start = time.time()
            seeds = getSeeds_sta(degree, i)
            if i == K - 1:
                time_end = time.time()
                print("time of DegreeMax:", time_end - time_start)
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    # print("DegreeMax", seeds)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def HDegree(df_hyper_matrix, K, R, beta, T):
    """
    HDegree algorithm
    """
    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(0, K):
            if i == K - 1:
                time_start = time.time()
            seeds = getSeeds_sta(degree, i)
            if i == K - 1:
                time_end = time.time()
                print("time of HDegree", time_end - time_start)
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    # print("HDegree", seeds)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def getDegreeList(degree):
    matrix = [np.arange(len(degree)), degree]
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    return df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)


def getMaxDegreeNode(degree, seeds):
    degree_copy = copy.deepcopy(degree)
    global chosenNode
    while 1:
        flag = 0
        degree_matrix = getDegreeList(degree_copy)
        node_index = degree_matrix.loc['node_index']
        for node in node_index:
            if node not in seeds:
                chosenNode = node
                flag = 1
                break
        if flag == 1:
            break
    return [chosenNode]


def updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds):
    edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]
    adj_set = []
    for edge in edge_set:
        adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
    adj_set_unique = np.unique(np.array(adj_set))
    for adj in adj_set_unique:
        adj_edge_set = np.where(df_hyper_matrix.loc[adj] == 1)[0]
        adj_adj_set = []
        for each in adj_edge_set:
            adj_adj_set.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
        if adj in adj_adj_set:
            adj_adj_set.remove(adj)
        sums = 0
        for adj_adj in adj_adj_set:
            if adj_adj in seeds:
                sums = sums + 1
        degree[adj] = degree[adj] - sums


def updateDeg_hsd(degree, chosenNode, df_hyper_matrix):
    edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]

    for edge in edge_set:
        node_set = np.where(df_hyper_matrix[edge] == 1)[0]
        for node in node_set:
            degree[node] = degree[node] - 1


def getDegreeWeighted(df_hyper_matrix, N):
    adj_matrix = np.dot(df_hyper_matrix, df_hyper_matrix.T)
    adj_matrix[np.eye(N, dtype=np.bool_)] = 0
    df_adj_matrix = pd.DataFrame(adj_matrix)
    return df_adj_matrix.sum(axis=1)


def getTotalAdj(df_hyper_matrix, N):
    deg_list = []
    nodes_arr = np.arange(N)
    for node in nodes_arr:
        node_list = []
        edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0]
        for edge in edge_set:
            node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
        node_set = np.unique(np.array(node_list))
        deg_list.append(len(list(node_set)) - 1)
    return np.array(deg_list)


def getSeeds_hdd(df_hyper_matrix, N, K, pre_seeds=None):
    seeds = []
    pre_num = 0
    if pre_seeds:
        seeds.extend(pre_seeds)
        pre_num = len(pre_seeds)
    degree = getTotalAdj(df_hyper_matrix, N)
    for j in tqdm(range(1, K + 1 - pre_num)):
        chosenNode = getMaxDegreeNode(degree, seeds)[0]
        seeds.append(chosenNode)
        updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
    return seeds


def getSeeds_hsd(df_hyper_matrix, N, K):
    seeds = []
    degree = getTotalAdj(df_hyper_matrix, N)
    for j in tqdm(range(1, K + 1)):
        chosenNode = getMaxDegreeNode(degree, seeds)[0]
        seeds.append(chosenNode)
        updateDeg_hsd(degree, chosenNode, df_hyper_matrix)
    return seeds


def hurDisc(df_hyper_matrix, inc_matrix, H, K, R, N, beta, T, pre_seeds=None, hurd_seeds=None):
    """
    HeuristicDegreeDiscount algorithm
    """
    if pre_seeds is None:
        pre_seeds = []
    inf_spread_matrix = []
    time_start = time.time()
    nodes = list(H.nodes())
    pre_seeds = [nodes.index(seed) for seed in pre_seeds]
    if hurd_seeds:
        seeds_list = hurd_seeds
    else:
        seeds_list = getSeeds_hdd(df_hyper_matrix, N, K, pre_seeds)
    A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    # node_incidence_nodes = {i: list(np.nonzero(A[i, :])[1]) for i in range(inc_matrix.shape[0])}
    # node_incidence_edges = {i: list(np.nonzero(inc_matrix[i, :])[1]) for i in range(inc_matrix.shape[0])}

    print(seeds_list)
    # node_HD = {n: [len(node_incidence_edges[nodes[n]]), len(node_incidence_nodes[nodes[n]])] for n in seeds_list}
    # print(node_HD)
    time_end = time.time()
    print("time of hurDisc:", time_end - time_start)
    print("hurDisc:", seeds_list)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, len(seeds_list)+1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            # print(seeds, ":", I_list)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def sglDisc(df_hyper_matrix, K, R, N, beta, T):
    """
    HeuristicSingleDiscount algorithm
    """
    inf_spread_matrix = []
    time_start = time.time()
    seeds_list = getSeeds_hsd(df_hyper_matrix, N, K)
    time_end = time.time()
    print("time of sglDisc:", time_end - time_start)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    print("HSDP:", seeds_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def HACE(df_hyper_matrix, inc_matrix, edges, H, K, R, pre_seeds, beta, T, alpha):
    """
    HeuristicSingleDiscount algorithm
    """
    A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    nodes = list(H.nodes())
    for i in range(A.shape[0]):
        A[i, i] = 0
    C = compute_C(np.array(inc_matrix.todense()))
    C = np.clip(C, 0, 1)
    inf_spread_matrix = []
    node_inc_node = {n: set(np.nonzero(A[n, :])[1]) for n in nodes}
    time_start = time.time()
    # seeds_list = seed_choose_v4_25(inc_matrix, H, edges, K)
    seeds_list = seeds_choose_v0630(inc_matrix, H, K, pre_seeds, beta, node_inc_node, alpha, C)
    print("time of HACE:", time.time() - time_start)
    # A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    # node_incidence_nodes = {i: list(np.nonzero(A[i, :])[1]) for i in range(inc_matrix.shape[0])}
    # node_incidence_edges = {i: list(np.nonzero(inc_matrix[i, :])[1]) for i in range(inc_matrix.shape[0])}
    # node_HD = {n: [len(node_incidence_edges[nodes[n]]), len(node_incidence_nodes[nodes[n]])] for n in seeds_list}
    # print(node_HD)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, len(seeds_list) + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            # print(seeds, ":", I_list)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    # print("SFVE:", seeds_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def generalGreedy(df_hyper_matrix, inc_matrix, H, K, R, beta, T):
    """
    GeneralGreedy algorithm
    """
    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    nodes = list(H.nodes())
    A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    node_incidence_nodes = {i: list(np.nonzero(A[i, :])[1]) for i in range(inc_matrix.shape[0])}
    node_incidence_edges = {i: list(np.nonzero(inc_matrix[i, :])[1]) for i in range(inc_matrix.shape[0])}
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        seeds = []
        for i in range(0, K):
            scale_list_temp = []
            maxNode = 0
            maxScale = 1
            for inode in range(0, len(degree)):
                if inode not in seeds:
                    seeds.append(inode)
                    scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
                    seeds.remove(inode)
                    scale_list_temp.append(scale)
                    if scale > maxScale:
                        maxNode = inode
                        maxScale = scale
            seeds.append(maxNode)
            scale_list.append(max(scale_list_temp))

        node_HD = {n: [len(node_incidence_edges[nodes[n]]), len(node_incidence_nodes[nodes[n]])] for n in seeds}
        print(node_HD)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def computeCI(L, N, df_hyper_matrix):
    CI_list = []
    degree = df_hyper_matrix.sum(axis=1)
    M = len(df_hyper_matrix.columns.values)
    for i in range(0, N):
        # 找到它的l阶邻居
        edge_set = np.where(df_hyper_matrix.loc[i] == 1)[0]
        if L == 1:
            node_list = []
            for edge in edge_set:
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
            if i in node_list:
                node_list.remove(i)
            node_set = np.unique(np.array(node_list))
        elif L == 2:
            node_list = []
            for edge in edge_set:
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
            if i in node_list:
                node_list.remove(i)
            node_set1 = np.unique(np.array(node_list))
            node_list2 = []
            edge_matrix = np.dot(df_hyper_matrix.T, df_hyper_matrix)
            edge_matrix[np.eye(M, dtype=np.bool_)] = 0
            df_edge_matrix = pd.DataFrame(edge_matrix)
            adj_edge_list = []
            for edge in edge_set:
                adj_edge_list.extend(list(np.where(df_edge_matrix[edge] != 0)[0]))
            adj_edge_set = np.unique(np.array(adj_edge_list))
            for each in adj_edge_set:
                node_list2.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
            node_set2 = list(np.unique(np.array(node_list2)))
            for node in node_set2:
                if node in list(node_set1):
                    # print(node_set2)
                    node_set2.remove(node)
            node_set = np.array(node_set2)
        ki = degree[i]
        sums = 0
        for u in node_set:
            sums = sums + (degree[u] - 1)
        CI_i = (ki - 1) * sums
        CI_list.append(CI_i)
    return CI_list


def getSeeds_ci(l, N, K, df_hyper_matrix):
    seeds = []
    n = np.ones(N)
    CI_list = computeCI(l, N, df_hyper_matrix)
    CI_arr = np.array(CI_list)
    for j in range(0, K):
        CI_chosen_val = CI_arr[np.where(n == 1)[0]]
        CI_chosen_index = np.where(n == 1)[0]
        index = np.where(CI_chosen_val == np.max(CI_chosen_val))[0][0]
        node = CI_chosen_index[index]
        n[node] = 0
        seeds.append(node)
    return seeds


def CIAgr(df_hyper_matrix, K, R, N, l, beta, T, ci_seeds=None):
    """
    H-CI algorithm
    """
    time_start = time.time()
    inf_spread_matrix = []
    if ci_seeds:
        seeds_list = ci_seeds
    else:
        seeds_list = getSeeds_ci(l, N, K, df_hyper_matrix)
    time_end = time.time()
    print(f"time of ci{l}:", time_end - time_start)
    print(f"ci{l}", seeds_list)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def getSeeds_ris(N, K, lamda, theta, df_hyper_matrix):
    S = []
    U = []
    # 迭代θ次
    for theta_iter in range(0, theta):
        df_matrix = copy.deepcopy(df_hyper_matrix)
        # 随机选择节点
        selected_node = random.sample(list(np.arange(len(df_hyper_matrix.index.values))), 1)[0]
        # 以1-λ的比例删边，构成子超图
        all_edges = np.arange(len(df_hyper_matrix.columns.values))
        prob = np.random.random(len(all_edges))
        index = np.where(prob > lamda)[0]
        for edge in index:
            df_matrix[edge] = 0
        # 将子超图映射到普通图
        adj_matrix = np.dot(df_matrix, df_matrix.T)
        adj_matrix[np.eye(N, dtype=np.bool_)] = 0
        df_adj_matrix = pd.DataFrame(adj_matrix)
        df_adj_matrix[df_adj_matrix > 0] = 1
        G = nx.from_numpy_array(df_adj_matrix.values)
        shortest_path = nx.shortest_path(G, target=selected_node)
        RR = []
        for each in shortest_path:
            RR.append(each)
        U.append(list(np.unique(np.array(RR))))
    # 重复k次
    for k in range(0, K):
        U_list = []
        for each in U:
            U_list.extend(each)
        dict = {}
        for each in U_list:
            if each in dict.keys():
                dict[each] = dict[each] + 1
            else:
                dict[each] = 1
        candidate_list = sorted(dict.items(), key=lambda item: item[1], reverse=True)
        chosen_node = candidate_list[0][0]
        S.append(chosen_node)
        for each in U:
            if chosen_node in each:
                U.remove(each)
    return S


def RISAgr(df_hyper_matrix, K, R, N, lamda, theta, beta, T, ris_seeds=None):
    """
    H-RIS algorithm
    """
    inf_spread_matrix = []
    time_start = time.time()
    if ris_seeds:
        seeds_list = ris_seeds
    else:
        seeds_list = getSeeds_ris(N, K, lamda, theta, df_hyper_matrix)
    time_end = time.time()
    print("time of RIS:", time_end - time_start)
    print("RIS", seeds_list)
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list


def UseSeedSet(df_hyper_matrix, K, R, seeds_list, beta, T):
    inf_spread_matrix = []
    for r in tqdm(range(R), desc="Loading..."):
        scale_list = []
        for i in range(1, K + 1):
            seeds = seeds_list[:i]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, beta, T)
            scale_list.append(scale)
        inf_spread_matrix.append(scale_list)
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    return final_scale_list
