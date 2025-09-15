import copy
import math
import random
import time

import networkx as nx
import numpy as np
from mpmath import sigmoid
from sklearn import preprocessing
from tqdm import tqdm

from IM.models import influence_SI


#######################################################################################################


def generalGreedy(H, R, k, weights, node_t, edge_t):
    """
    GeneralGreedy algorithm
    """
    inf_spread_matrix = []
    for _ in tqdm(range(R), desc="Loading..."):
        scale_list = []
        seeds_set = []
        for i in tqdm(range(0, k)):
            scale_list_temp = []
            maxNode = 0
            maxScale = 1
            for inode in range(0, H.shape[0]):
                if inode not in seeds_set:
                    seeds_set.append(inode)
                    scale = influence_SI(seeds_set, H)
                    seeds_set.remove(inode)
                    scale_list_temp.append(scale)
                    if scale > maxScale:
                        maxNode = inode
                        maxScale = scale

            seeds_set.append(maxNode)
            scale_list.append(max(scale_list_temp))
            print(_, i)

        print("seeds_greedy: ", seeds_set)
        inf_spread_matrix.append(scale_list)

    final_scale_list = np.array(inf_spread_matrix).mean(axis=0).tolist()
    return final_scale_list


def H_Degree(H, num):
    node_degree = np.sum(H, axis=1).reshape(-1).tolist()[0]
    node_degree_dict = {i: node_degree[i] for i in range(len(node_degree))}
    node_degree_sorted = sorted(node_degree_dict.items(), key=lambda x: x[1], reverse=True)
    return [node_degree_sorted[i][0] for i in range(num)]


def get_A_degree(H):
    """
    return a dict {node: degree} sorted based on degree
    """
    H_degree = np.sum(H, axis=1).reshape(-1).tolist()[0]
    D = np.diag(H_degree)
    A = H @ H.T - D
    A[A > 0] = 1
    degree = np.sum(A, axis=1).reshape(-1).tolist()[0]
    N = len(degree)
    A_degree_dict = {i: degree[i] for i in range(N)}
    A_degree_sorted = sorted(A_degree_dict.items(), key=lambda x: x[1], reverse=True)
    A_degree_dict = {A_degree_sorted[i][0]: A_degree_sorted[i][1] for i in range(len(A_degree_sorted))}
    return A_degree_dict


def getMaxDegreeNode(degree: dict, seeds):
    degree_copy = copy.deepcopy(degree)
    chosenNode = 0
    for node in degree_copy.keys():
        if node not in seeds:
            chosenNode = node
            break

    return chosenNode


def getSeeds_hdd(H, K):
    seeds = []
    degree_dict = get_A_degree(H)
    for _ in range(K):
        chosenNode = getMaxDegreeNode(degree_dict, seeds)
        seeds.append(chosenNode)
        degree_dict = updateDeg_hur(degree_dict, chosenNode, H, seeds)

    return seeds


def getSeeds_hsd(H, K):
    seeds = []
    degree_dict = get_A_degree(H)
    for _ in range(K):
        chosenNode = getMaxDegreeNode(degree_dict, seeds)
        seeds.append(chosenNode)
        degree_dict = updateDeg_hsd(degree_dict, chosenNode, H)
    return seeds


def updateDeg_hur(degree, chosenNode, H, seeds):
    degree = copy.deepcopy(degree)
    edge_of_node = np.nonzero(H[chosenNode, :])[1]
    adj_nodes = set()
    for edge in edge_of_node:
        adj_nodes |= set(np.nonzero(H[:, edge])[0])

    H_degree = np.sum(H, axis=1).reshape(-1).tolist()[0]
    D = np.diag(H_degree)
    A = H @ H.T - D
    A[A > 0] = 1
    for node in adj_nodes:
        adj_nodes = np.nonzero(A[node, :])[1]
        updated_adj = set(adj_nodes) - set(seeds)
        degree[node] = len(updated_adj)

    return degree


def updateDeg_hsd(degree, chosenNode, H):
    degree = copy.deepcopy(degree)
    edge_set = np.nonzero(H[chosenNode, :])[1]

    for edge in edge_set:
        node_set = np.nonzero(H[:, edge])[0]
        for node in node_set:
            degree[node] -= 1

    return degree


def DegreeMax(H, K):
    """
    Degree algorithm
    """
    degree_dict = get_A_degree(H)
    node_sorted_degree = list(degree_dict.keys())
    return node_sorted_degree[:K]


def line_graph(inc_matrix):
    return np.dot(inc_matrix.T, inc_matrix)


def strength(A):
    s_res = []
    num_node = A.shape[0]
    for k in range(num_node):
        s_res.append(np.sum(A.getrow(k).todense()) - A[k, k])
    s_res /= sum(s_res)
    s_res = [(k, s_res[k]) for k in range(num_node)]
    return s_res

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

def HACE_all(inc_matrix, H, seeds_num, pre_seeds, beta, node_inc_node, C, T):
    time_start = time.time()
    A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    for i in range(A.shape[0]):
        A[i, i] = 0

    node_set = list(H.nodes())
    node_for_choose = set(H.nodes())
    seeds = []
    pre_num = 0
    if pre_seeds:
        seeds.extend(pre_seeds)
        pre_num = len(pre_seeds)
    node_for_choose -= set(seeds)
    contacted_prob = {n: 0. for n in node_for_choose}
    N_s = set()
    times = []
    for _ in tqdm(range(seeds_num - pre_num)):
        max_cannot_infed_inf, max_cannot_infed_n = 0, -1
        max_rou, max_rou_n = 0, 0
        max_dcc = 0
        for n in node_for_choose:
            inf = sum([C[node, n] * C[node, n] / (C[node, n] + contacted_prob[node]) for node in (node_inc_node[n]-set(seeds))])
            infed = contacted_prob[n]
            if infed == 0:
                if inf > max_cannot_infed_inf:
                    max_cannot_infed_inf = inf
                    max_cannot_infed_n = n
            elif inf / infed > max_rou:
                max_rou = inf / infed
                max_rou_n = n
                max_dcc = inf * (1-infed*beta)

        seed = max_rou_n
        if max_cannot_infed_n != -1 and max_dcc < max_cannot_infed_inf:
            seed = max_cannot_infed_n
        seeds.append(seed)
        for N_n in node_inc_node[seed]-set(seeds):
            contacted_prob[N_n] += C[N_n, seed]
        node_for_choose.remove(seed)
        N_s |= node_inc_node[seed]
        times.append(time.time() - time_start)
    seeds = [node_set.index(i) for i in seeds]
    return seeds, times


def HCLI(inc_matrix, H, seeds_num, pre_seeds, beta, node_inc_node, alpha, C, T):
    time_start = time.time()
    node_set = list(H.nodes())
    node_for_choose = set(H.nodes())
    seeds = []
    pre_num = 0
    node_for_choose -= set(seeds)
    contacted_prob = {n: 0. for n in node_for_choose}
    N_s = set()
    X = [0] * inc_matrix.shape[0]
    times = []

    for _ in tqdm(range(seeds_num - pre_num)):
        max_rou, max_rou_n = -1, 0
        for n in node_for_choose:
            ACC = sum([C[node, n] * C[node, n] / (C[node, n] + contacted_prob[node]) for node in (node_inc_node[n]-set(seeds))])
            rou = ACC*(1-X[n])
            if rou > max_rou:
                max_rou = rou
                max_rou_n = n

        seed = max_rou_n
        X[seed] = 1
        seeds.append(seed)
        inc_nodes = node_inc_node[seed] - set(seeds)
        N_s |= inc_nodes
        X_temp = [0] * len(X)
        for N_n in seeds:
            X_temp[N_n] = 1
            X[N_n] = 1
        for N_n in set(node_set) - set(seeds):
            if N_n in inc_nodes:
                contacted_prob[N_n] = C[N_n, seed]+contacted_prob[N_n]
            temp_not_infected = 1.
            for j in node_inc_node[N_n]-set(seeds):
                temp_not_infected = temp_not_infected * (1-beta * C[N_n, j] * X[j])
            temp_not_infected = temp_not_infected * (1 - beta * C[N_n, seed])
            X_temp[N_n] = X[N_n]+(1 - temp_not_infected)*(1-X[N_n])
            if X_temp[N_n] > 1:
                X_temp[N_n] = 1

        X = X_temp
        node_for_choose.remove(seed)
        times.append(time.time()-time_start)

    seeds = [node_set.index(i) for i in seeds]
    return seeds, times

