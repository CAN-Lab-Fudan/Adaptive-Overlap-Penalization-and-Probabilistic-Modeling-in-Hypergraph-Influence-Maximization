import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Hyperspreading import Hyperspreading
from tqdm import tqdm
import copy
import random

from algorithm import getTotalAdj
from transform import Transform
from copy import deepcopy
import networkx as nx
import matplotlib
import datetime
import heapq
from queue import Queue
from collections import deque
import math

hs = Hyperspreading()
tf = Transform()


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # 可以选择"w"
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal


def compute_2hop_p(beta1, beta2, T):
    p = 0
    if beta1 == 1 and beta2 != 1:
        for i in range(2, T + 1):
            for j in range(1, i):
                p += math.pow(1 - beta2, i - 1 - j) * beta2

    if beta1 != 1 and beta2 == 1:
        for i in range(2, T + 1):
            p += math.pow(1 - beta1, i - 1) * beta1

    if beta1 == 1 and beta2 == 1:
        p = 1

    if beta1 != 1 and beta2 != 1:
        for i in (range(2, T + 1)):
            for j in range(1, i):
                # p += math.pow(1-beta1, j-1) * beta1 * math.pow(1-beta2, i-1-j) * beta2
                p += math.pow(1 - beta1, j - 1) * math.pow(1 - beta2, i - 1 - j)

    return p * beta1 * beta2


def compute_3hop_p(beta1, beta2, beta3, T):
    p = 0
    if beta1 != 1 and beta2 != 1 and beta3 != 1:
        for j in (range(3, T + 1)):
            for i in range(2, j):
                for k in range(1, i):
                    p += math.pow(1 - beta1, k - 1) * beta1 * math.pow(1 - beta2, i - k - 1) * beta2 * math.pow(
                        1 - beta3, j - i - 1) * beta3
    return p


def compute_1hop_inf(fileName, df_hyper_matrix, N, beta, T):
    bi_adj, probability_mx, neighbour_dict, edge_set, edge_prob = Hyper_to_direct(df_hyper_matrix)
    probability_mx = probability_mx * beta
    _1hop_psum = np.array([])
    dic = {}

    for inode in (range(N)):  # i -> j -> k
        i_1hop_psum = 0
        for jnode in neighbour_dict[inode]:
            beta1 = probability_mx[inode][jnode]
            if str(beta1) in dic:
                _1hop_p = dic[str(beta1)]
            else:
                _1hop_p = (1 - math.pow(1 - beta1, T))
                dic[str(beta1)] = _1hop_p
            i_1hop_psum += _1hop_p
        _1hop_psum = np.append(_1hop_psum, i_1hop_psum)


    return _1hop_psum, probability_mx, neighbour_dict, dic


def getNeighbour(targetNode, df_hyper_matrix):
    """
    :param targetNode:
    :param df_hyper_matrix:
    :return: the neighbours of the targetNode
    """
    edge_set = np.where(df_hyper_matrix.loc[targetNode] == 1)[0]  # chosenNode所在的Hyperedge集合
    adj_set = []
    for edge in edge_set:
        adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
    adj_set_unique = np.unique(np.array(adj_set))  # adj_set_unique为chosenNode的邻居集合(包含自身)
    adj_set_unique = list(adj_set_unique)
    if targetNode in adj_set_unique:
        adj_set_unique.remove(targetNode)
    return adj_set_unique  # list


def getProbability(sourceNode, targetNode, Hdegree, df_hyper_matrix):
    """
    compute the probability p(u,v) node u activates node v
    :param sourceNode: u
    :param targetNode: v
    :param Hdegree: hyper-degree list
    :param df_hyper_matrix: incidence matrix
    :return: p(u,v)
    """
    edge_set1 = np.where(df_hyper_matrix.loc[sourceNode] == 1)[0]  # Node所在的Hyperedge集合
    edge_set2 = np.where(df_hyper_matrix.loc[targetNode] == 1)[0]  # Node所在的Hyperedge集合
    inter = np.intersect1d(edge_set1, edge_set2)  # 求交集
    num_common_edges = len(inter)
    p = num_common_edges / Hdegree[sourceNode]
    return p


def Hyper_to_direct(df_hyper_matrix):
    """
    transform a hypergraph under SICP model to a directed ordinary graph
    :param df_hyper_matrix:
    """
    node_num = df_hyper_matrix.shape[0]
    bi_adj = np.zeros((node_num, node_num))  # binary format of adjacency matrix
    edge_set = np.array([], dtype=int)
    edge_prob = np.array([])
    HyperDegree = df_hyper_matrix.sum(axis=1)
    Hdegree = np.array([])
    for item in HyperDegree:
        Hdegree = np.append(Hdegree, item)

    probability_mx = np.zeros((node_num, node_num), dtype=float)
    neighbour_dict = {}
    for inode in range(node_num):
        neighbour_dict[inode] = getNeighbour(inode, df_hyper_matrix)
        for jnode in neighbour_dict[inode]:
            bi_adj[inode][jnode] = 1
            probability_mx[inode][jnode] = getProbability(inode, jnode, Hdegree, df_hyper_matrix)
            edge_set = np.append(edge_set, [inode, jnode])
            edge_prob = np.append(edge_prob, probability_mx[inode][jnode])
    edge_set = edge_set.reshape([-1, 2])

    return bi_adj, probability_mx, neighbour_dict, edge_set, edge_prob


def compute_1hop_inf_NearExact(fileName, df_hyper_matrix, N, beta, T):
    bi_adj, probability_mx, neighbour_dict, edge_set, edge_prob = Hyper_to_direct(df_hyper_matrix)
    probability_mx = probability_mx * beta
    _1hop_psum = np.array([])
    dic = {}

    for inode in tqdm(range(N)):  # i -> j -> k
        i_1hop_psum = 0
        for jnode in neighbour_dict[inode]:
            beta1 = probability_mx[inode][jnode]
            if str(beta1) in dic:
                _1hop_p = dic[str(beta1)]
            else:
                _1hop_p = (1 - math.pow(1 - beta1, T))
                dic[str(beta1)] = _1hop_p
            i_1hop_psum += _1hop_p
        _1hop_psum = np.append(_1hop_psum, i_1hop_psum)

    return _1hop_psum, probability_mx, neighbour_dict, dic


def compute_2hop_inf_NearExact(fileName, df_hyper_matrix, N, beta, T):
    bi_adj, probability_mx, neighbour_dict, edge_set, edge_prob = Hyper_to_direct(df_hyper_matrix)
    probability_mx = probability_mx * beta
    dic1 = {}
    dic2 = {}

    psum = np.array([])
    for inode in tqdm(range(N)):  # i -> j -> k
        dic_path_prob = {}
        stack = deque()  # 栈
        for jnode in neighbour_dict[inode]:
            stack.append(jnode)

        while len(stack) != 0:
            jnode = stack.pop()
            beta1 = probability_mx[inode][jnode]
            if str(beta1) not in dic1:
                dic1[str(beta1)] = math.pow(1 - beta1, T)
            i_1hop = dic1[str(beta1)]
            if jnode not in dic_path_prob:
                dic_path_prob[jnode] = np.array([i_1hop])
            else:
                dic_path_prob[jnode] = np.append(dic_path_prob[jnode], i_1hop)

            for knode in neighbour_dict[jnode]:
                if knode != inode:
                    beta2 = probability_mx[jnode][knode]
                    if str(beta1) + "_" + str(beta2) in dic2:
                        _2hop_p = dic2[str(beta1) + "_" + str(beta2)]
                    else:
                        _2hop_p = compute_2hop_p(beta1, beta2, T)
                        dic2[str(beta1) + "_" + str(beta2)] = _2hop_p
                    if knode not in dic_path_prob:
                        dic_path_prob[knode] = np.array([1 - _2hop_p])
                    else:
                        dic_path_prob[knode] = np.append(dic_path_prob[knode], 1 - _2hop_p)

        p_path_fail = np.array([])
        for i in dic_path_prob:
            p_path_fail = np.append(p_path_fail, np.prod(dic_path_prob[i]))
        p_path_success = 1 - p_path_fail
        psum = np.append(psum, np.sum(p_path_success))
    _2hop_psum = psum + 1
    return _2hop_psum, probability_mx, neighbour_dict, dic1, dic2


def compute_3hop_inf_NearExact(fileName, df_hyper_matrix, N, beta, T):
    bi_adj, probability_mx, neighbour_dict, edge_set, edge_prob = Hyper_to_direct(df_hyper_matrix)
    probability_mx = probability_mx * beta
    dic_1hop = {}
    dic_2hop = {}
    dic_3hop = {}

    psum = np.array([])
    for inode in tqdm(range(N)):  # i -> j -> k -> l
        dic_path_prob = {}

        stack1 = deque()  # 栈
        for jnode in neighbour_dict[inode]:
            stack1.append(jnode)

        while len(stack1) != 0:
            jnode = stack1.pop()
            beta1 = probability_mx[inode][jnode]
            if str(beta1) not in dic_1hop:
                dic_1hop[str(beta1)] = math.pow(1 - beta1, T)
            i_1hop = dic_1hop[str(beta1)]
            if jnode not in dic_path_prob:
                dic_path_prob[jnode] = np.array([i_1hop])
            else:
                dic_path_prob[jnode] = np.append(dic_path_prob[jnode], i_1hop)

            for knode in neighbour_dict[jnode]:
                if knode != inode:
                    beta2 = probability_mx[jnode][knode]
                    if str(beta1) + "_" + str(beta2) in dic_2hop:
                        _2hop_p = dic_2hop[str(beta1) + "_" + str(beta2)]
                    else:
                        _2hop_p = compute_2hop_p(beta1, beta2, T)
                        dic_2hop[str(beta1) + "_" + str(beta2)] = _2hop_p
                    if knode not in dic_path_prob:
                        dic_path_prob[knode] = np.array([1 - _2hop_p])
                    else:
                        dic_path_prob[knode] = np.append(dic_path_prob[knode], 1 - _2hop_p)

                    for lnode in neighbour_dict[knode]:
                        if lnode != jnode and lnode != inode:
                            beta3 = probability_mx[knode][lnode]
                            if str(beta1) + "_" + str(beta2) + "_" + str(beta3) in dic_3hop:
                                _3hop_p = dic_3hop[str(beta1) + "_" + str(beta2) + "_" + str(beta3)]
                            else:
                                _3hop_p = compute_3hop_p(beta1, beta2, beta3, T)
                                dic_3hop[str(beta1) + "_" + str(beta2) + "_" + str(beta3)] = _3hop_p
                            if lnode not in dic_path_prob:
                                dic_path_prob[lnode] = np.array([1 - _3hop_p])
                            else:
                                dic_path_prob[lnode] = np.append(dic_path_prob[lnode], 1 - _3hop_p)

        p_path_fail = np.array([])
        for i in dic_path_prob:
            p_path_fail = np.append(p_path_fail, np.prod(dic_path_prob[i]))
        p_path_success = 1 - p_path_fail
        psum = np.append(psum, np.sum(p_path_success))

    _3hop_psum = psum + 1
    return _3hop_psum, probability_mx, neighbour_dict, dic_1hop, dic_2hop, dic_3hop


def ConstrainedDFS(targetNode, N, df_hyper_matrix, L):
    nbr_dic = {}
    for i in range(N):
        nbr_dic[i] = getNeighbour(i, df_hyper_matrix)

    stack = deque()
    pt = deque()
    pt.append(targetNode)
    psum = 0
    END = False
    while len(pt) != 0:
        if END == False:
            if len(pt) <= L:
                v_j = pt[-1]
                stack.append('&')
                for v_k in nbr_dic[v_j]:
                    if v_k not in pt:
                        stack.append(v_k)
            else:
                pt.pop()

        node = stack.pop()
        END = False
        if node != '&':
            pt.append(node)
            psum += 1
            # print("pt:", pt)
            print("pt:", pt, "\tstack:", stack)
        else:
            pt.pop()
            END = True

    return 0


def adeff(fileName, df_hyper_matrix, K, R, N, beta, T):
    """
    Adaptive Neighbourhood Coefficient Algorithm
    :param df_hyper_matrix:
    :param K:
    :param R:
    :param N:
    :return:
    """
    time_start = time.time()
    seeds = []
    new_df_hyper_matrix = deepcopy(df_hyper_matrix)
    # sys.stdout = Logger('time_'+fileName + '_adaptive_coeff.txt')
    times = []
    for i in (range(K)):
        degree = getTotalAdj(new_df_hyper_matrix, N)
        HyperDegree = df_hyper_matrix.sum(axis=1)
        Hdegree = np.array([])
        for item in HyperDegree:
            if item == 0:
                item = 1
            Hdegree = np.append(Hdegree, item)

        coeff = degree / Hdegree
        maxNode = np.argmax(coeff)
        seeds.append(maxNode)

        new_df_hyper_matrix.loc[maxNode] = 0  # remove maxNode from the original hypergraph
        """find the nodes infected by maxNode many times"""
        dict_times = {}
        scale_list_temp = []
        for j in range(200):
            tmp_seeds = [maxNode]
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
            for node in I_list:
                if node not in dict_times:
                    dict_times[node] = 1
                else:
                    dict_times[node] += 1
        k = int(sum(scale_list_temp) / len(scale_list_temp))
        main_infected_nodes = heapq.nlargest(k, dict_times, key=dict_times.get)
        """remove the nodes infected by maxNode"""
        for node in main_infected_nodes:
            new_df_hyper_matrix.loc[node] = 0
        times.append(time.time()-time_start)
    # sys.stdout.reset()
    cur_time = time.time()
    run_time = cur_time - time_start
    print('adeff', run_time)
    return [], [], times

    """compute influence(scale) of seeds"""
    scale_list = []
    for i in tqdm(range(1, len(seeds) + 1)):
        tmp_seeds = seeds[0:i]
        # print(i, ": tmp_seeds:", tmp_seeds)
        scale_list_temp = []
        for j in range(R):
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
        tmp_scale = sum(scale_list_temp) / len(scale_list_temp)
        scale_list.append(tmp_scale)
    return np.array(seeds), np.array(scale_list), times


def adaptive_1hopinf(fileName, df_hyper_matrix, K, R, N, beta, T):
    starttime = datetime.datetime.now()
    seeds = []
    new_df_hyper_matrix = deepcopy(df_hyper_matrix)
    for i in tqdm(range(K)):
        _1hop_inf, probability_mx, neighbour_dict, dic = compute_1hop_inf(fileName, new_df_hyper_matrix, N, beta, T)

        maxNode = np.argmax(_1hop_inf)
        seeds.append(maxNode)
        new_df_hyper_matrix.loc[maxNode] = 0  # remove maxNode from the original hypergraph

        """find the nodes infected by maxNode many times"""
        dict_times = {}
        scale_list_temp = []
        for j in range(200):
            tmp_seeds = [maxNode]
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
            for node in I_list:
                if node not in dict_times:
                    dict_times[node] = 1
                else:
                    dict_times[node] += 1
        k = int(sum(scale_list_temp) / len(scale_list_temp))
        main_infected_nodes = heapq.nlargest(k, dict_times, key=dict_times.get)

        """remove the nodes infected by maxNode"""
        for node in main_infected_nodes:
            new_df_hyper_matrix.loc[node] = 0

    endtime = datetime.datetime.now()
    print("beta=", beta, "T=", T, "-------------- adaptive_1hopinf time on :" + fileName, endtime - starttime)

    """compute influence(scale) of seeds"""
    scale_list = []
    for i in tqdm(range(1, len(seeds) + 1)):
        tmp_seeds = seeds[0:i]
        # print(i, ": tmp_seeds:", tmp_seeds)
        scale_list_temp = []
        for j in range(R):
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
        tmp_scale = sum(scale_list_temp) / len(scale_list_temp)
        scale_list.append(tmp_scale)
    return np.array(seeds), np.array(scale_list)


def _1hopinf_NearExact_topK(fileName, df_hyper_matrix, K, R, N, beta, T):
    new_df_hyper_matrix = deepcopy(df_hyper_matrix)
    _1hop_inf, probability_mx, neighbour_dict, dic = compute_1hop_inf(fileName, new_df_hyper_matrix, N, beta, T)
    seeds = _1hop_inf.argsort()[-K:][::-1]  # 获取coeff最大的前k个索引

    """compute influence(scale) of seeds"""
    scale_list = []
    for i in tqdm(range(1, len(seeds) + 1)):
        tmp_seeds = seeds[0:i]
        # print(i, ": tmp_seeds:", tmp_seeds)
        scale_list_temp = []
        for j in range(R):
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
        tmp_scale = sum(scale_list_temp) / len(scale_list_temp)
        scale_list.append(tmp_scale)
    return np.array(seeds), np.array(scale_list)


def MIE_1(fileName, df_hyper_matrix, K, R, N, beta, T):
    """
    The multi-hop influence estimation algorithm, L=1
    returns: seeds: the seed node set detected by MIE(L=1), scale_list: the influence spread of seed node set.
    """
    start_time = time.time()
    seeds = []
    new_df_hyper_matrix = deepcopy(df_hyper_matrix)
    _1hop_inf, probability_mx, neighbour_dict, dic = compute_1hop_inf(fileName, new_df_hyper_matrix, N, beta, T)
    # print(_1hop_inf, probability_mx, neighbour_dict, dic)

    # sys.stdout = Logger('efficiency//time_' + fileName + '_adaptive_1hopinf_NearExact.txt')
    times = []
    for i in tqdm(range(K)):
        maxNode = np.argmax(_1hop_inf)
        seeds.append(maxNode)
        # cur_time = time.time()
        # run_time = cur_time - start_time
        # print(run_time)

        """find the nodes infected by maxNode many times"""
        dict_times = {}
        scale_list_temp = []
        for j in range(200):
            tmp_seeds = [maxNode]
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
            for node in I_list:
                if node not in dict_times:
                    dict_times[node] = 1
                else:
                    dict_times[node] += 1
        k = int(sum(scale_list_temp) / len(scale_list_temp))
        main_infected_nodes = heapq.nlargest(k, dict_times, key=dict_times.get)

        """remove the nodes infected by maxNode"""
        for nbr in neighbour_dict[int(maxNode)]:
            if maxNode in neighbour_dict[nbr]:
                neighbour_dict[nbr].remove(maxNode)
        neighbour_dict[int(maxNode)] = []
        new_df_hyper_matrix.loc[maxNode] = 0  # remove maxNode from the original hypergraph.

        for fl in main_infected_nodes:
            for nbr in neighbour_dict[fl]:
                if fl in neighbour_dict[nbr]:
                    neighbour_dict[nbr].remove(fl)
            new_df_hyper_matrix.loc[fl] = 0
            neighbour_dict[fl] = []

        node_alone = int(-1)
        for j in range(new_df_hyper_matrix.shape[1]):
            if np.sum(new_df_hyper_matrix[j]) == 1:
                node_alone = int(np.where(new_df_hyper_matrix[j] == 1)[0])

        if node_alone != -1:
            for node in neighbour_dict[node_alone]:
                edge_set1 = np.where(new_df_hyper_matrix.loc[node_alone] == 1)[0]  # node_alone所在的Hyperedge集合
                edge_set2 = np.where(new_df_hyper_matrix.loc[node] == 1)[0]  # node所在的Hyperedge集合
                inter = np.intersect1d(edge_set1, edge_set2)  # 求交集
                num_common_edges = len(inter)
                p = num_common_edges / len(edge_set1)
                probability_mx[node_alone][node] = p * beta

        """Update _1hop_inf"""
        _1hop_psum = np.array([])
        for inode in (range(N)):  # i -> j -> k
            i_1hop_psum = 0
            for jnode in neighbour_dict[inode]:
                beta1 = probability_mx[inode][jnode]
                if str(beta1) in dic:
                    _1hop_p = dic[str(beta1)]
                else:
                    _1hop_p = (1 - math.pow(1 - beta1, T))
                    dic[str(beta1)] = _1hop_p
                i_1hop_psum += _1hop_p
            _1hop_psum = np.append(_1hop_psum, i_1hop_psum)
        _1hop_inf = _1hop_psum
        times.append(time.time()-start_time)

    # sys.stdout.reset()
    # endtime = datetime.datetime.now()
    # print("beta=", beta, "T=", T, "-------------- adaptive_1hopinf_fast time on :" + fileName, endtime - starttime)
    cur_time = time.time()
    run_time = cur_time - start_time
    print("MIE(l=1)", run_time)
    """compute influence(scale) of seeds"""
    # scale_list = []
    # for i in tqdm(range(1, len(seeds) + 1)):
    #     tmp_seeds = seeds[0:i]
    #     # print(i, ": tmp_seeds:", tmp_seeds)
    #     scale_list_temp = []
    #     for j in range(R):
    #         scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
    #         scale_list_temp.append(scale_temp)
    #     tmp_scale = sum(scale_list_temp) / len(scale_list_temp)
    #     scale_list.append(tmp_scale)
    return seeds, times


def MIE_2(fileName, df_hyper_matrix, K, R, N, beta, T):
    time_start = time.time()
    start_time = time.time()
    seeds = []
    new_df_hyper_matrix = deepcopy(df_hyper_matrix)
    _2hop_psum, probability_mx, neighbour_dict, dic1, dic2 = compute_2hop_inf_NearExact(fileName, df_hyper_matrix, N,
                                                                                        beta, T)

    # sys.stdout = Logger('efficiency//time_'+fileName + '_adaptive_2hopinf_NearExact.txt')
    """get seed nodes iteratively"""
    times = []
    for i in tqdm(range(K)):
        maxNode = np.argmax(_2hop_psum)
        seeds.append(maxNode)

        """Find the followers of maxNode"""
        dict_times = {}
        scale_list_temp = []
        for j in range(200):
            tmp_seeds = [maxNode]
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
            for node in I_list:
                if node not in dict_times:
                    dict_times[node] = 1
                else:
                    dict_times[node] += 1
        k = int(sum(scale_list_temp) / len(scale_list_temp))
        main_infected_nodes = heapq.nlargest(k, dict_times, key=dict_times.get)

        """Remove maxNode and its followers"""
        new_df_hyper_matrix.loc[maxNode] = 0
        for node in neighbour_dict[int(maxNode)]:
            if maxNode in neighbour_dict[node]:
                neighbour_dict[node].remove(maxNode)
        neighbour_dict[int(maxNode)] = []

        for node in main_infected_nodes:
            new_df_hyper_matrix.loc[node] = 0
            for rnode in neighbour_dict[node]:
                if node in neighbour_dict[rnode]:
                    neighbour_dict[rnode].remove(node)
            neighbour_dict[node] = []

        node_alone = int(-1)
        for j in range(new_df_hyper_matrix.shape[1]):
            if np.sum(new_df_hyper_matrix[j]) == 1:
                node_alone = int(np.where(new_df_hyper_matrix[j] == 1)[0])

        if node_alone != -1:
            for node in neighbour_dict[node_alone]:
                edge_set1 = np.where(new_df_hyper_matrix.loc[node_alone] == 1)[0]  # node_alone所在的Hyperedge集合
                edge_set2 = np.where(new_df_hyper_matrix.loc[node] == 1)[0]  # node所在的Hyperedge集合
                inter = np.intersect1d(edge_set1, edge_set2)  # 求交集
                num_common_edges = len(inter)
                p = num_common_edges / len(edge_set1)
                probability_mx[node_alone][node] = p * beta

        """Update _2hop_psum"""
        psum = np.array([])
        for inode in range(N):  # i -> j -> k
            dic_path_prob = {}
            stack = deque()  # 栈
            for jnode in neighbour_dict[inode]:
                stack.append(jnode)

            while len(stack) != 0:
                jnode = stack.pop()
                beta1 = probability_mx[inode][jnode]
                if str(beta1) not in dic1:
                    dic1[str(beta1)] = math.pow(1 - beta1, T)
                i_1hop = dic1[str(beta1)]
                if jnode not in dic_path_prob:
                    dic_path_prob[jnode] = np.array([i_1hop])
                else:
                    dic_path_prob[jnode] = np.append(dic_path_prob[jnode], i_1hop)

                for knode in neighbour_dict[jnode]:
                    if knode != inode:
                        beta2 = probability_mx[jnode][knode]
                        if str(beta1) + "_" + str(beta2) in dic2:
                            _2hop_p = dic2[str(beta1) + "_" + str(beta2)]
                        else:
                            _2hop_p = compute_2hop_p(beta1, beta2, T)
                            dic2[str(beta1) + "_" + str(beta2)] = _2hop_p
                        if knode not in dic_path_prob:
                            dic_path_prob[knode] = np.array([1 - _2hop_p])
                        else:
                            dic_path_prob[knode] = np.append(dic_path_prob[knode], 1 - _2hop_p)

            p_path_fail = np.array([])
            for i in dic_path_prob:
                p_path_fail = np.append(p_path_fail, np.prod(dic_path_prob[i]))
            p_path_success = 1 - p_path_fail
            psum = np.append(psum, np.sum(p_path_success))
        _2hop_psum = psum + 1
        times.append(time.time()-time_start)

    # sys.stdout.reset()
    cur_time = time.time()
    run_time = cur_time - start_time
    print("MIE(l=2)", run_time)
    # endtime = datetime.datetime.now()
    # print("beta=", beta, "T=", T, "-------------- adaptive_2hopinf_NearExact time on :" + fileName, endtime - starttime)

    return seeds, times


def MIE_3(fileName, df_hyper_matrix, K, R, N, beta, T):
    starttime = datetime.datetime.now()
    seeds = []
    new_df_hyper_matrix = deepcopy(df_hyper_matrix)
    _3hop_psum, probability_mx, neighbour_dict, dic_1hop, dic_2hop, dic_3hop = compute_3hop_inf_NearExact(fileName,
                                                                                                          df_hyper_matrix,
                                                                                                          N, beta, T)

    """get seed nodes iteratively"""
    for i in tqdm(range(K)):
        maxNode = np.argmax(_3hop_psum)
        seeds.append(maxNode)

        """Find the followers of maxNode"""
        dict_times = {}
        scale_list_temp = []
        for j in range(200):
            tmp_seeds = [maxNode]
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
            for node in I_list:
                if node not in dict_times:
                    dict_times[node] = 1
                else:
                    dict_times[node] += 1
        k = int(sum(scale_list_temp) / len(scale_list_temp))
        main_infected_nodes = heapq.nlargest(k, dict_times, key=dict_times.get)

        """Remove maxNode and its followers"""
        new_df_hyper_matrix.loc[maxNode] = 0
        for node in neighbour_dict[int(maxNode)]:
            if maxNode in neighbour_dict[node]:
                neighbour_dict[node].remove(maxNode)
        neighbour_dict[int(maxNode)] = []

        for node in main_infected_nodes:
            new_df_hyper_matrix.loc[node] = 0
            for rnode in neighbour_dict[node]:
                if node in neighbour_dict[rnode]:
                    neighbour_dict[rnode].remove(node)
            neighbour_dict[node] = []

        node_alone = int(-1)
        for j in range(new_df_hyper_matrix.shape[1]):
            if np.sum(new_df_hyper_matrix[j]) == 1:
                node_alone = int(np.where(new_df_hyper_matrix[j] == 1)[0])

        if node_alone != -1:
            for node in neighbour_dict[node_alone]:
                edge_set1 = np.where(new_df_hyper_matrix.loc[node_alone] == 1)[0]  # node_alone所在的Hyperedge集合
                edge_set2 = np.where(new_df_hyper_matrix.loc[node] == 1)[0]  # node所在的Hyperedge集合
                inter = np.intersect1d(edge_set1, edge_set2)  # 求交集
                num_common_edges = len(inter)
                p = num_common_edges / len(edge_set1)
                probability_mx[node_alone][node] = p * beta

        """Update _3hop_psum"""
        psum = np.array([])
        for inode in tqdm(range(N)):  # i -> j -> k -> l
            dic_path_prob = {}

            stack1 = deque()  # 栈
            for jnode in neighbour_dict[inode]:
                stack1.append(jnode)

            while len(stack1) != 0:
                jnode = stack1.pop()
                beta1 = probability_mx[inode][jnode]
                if str(beta1) not in dic_1hop:
                    dic_1hop[str(beta1)] = math.pow(1 - beta1, T)
                i_1hop = dic_1hop[str(beta1)]
                if jnode not in dic_path_prob:
                    dic_path_prob[jnode] = np.array([i_1hop])
                else:
                    dic_path_prob[jnode] = np.append(dic_path_prob[jnode], i_1hop)

                for knode in neighbour_dict[jnode]:
                    if knode != inode:
                        beta2 = probability_mx[jnode][knode]
                        if str(beta1) + "_" + str(beta2) in dic_2hop:
                            _2hop_p = dic_2hop[str(beta1) + "_" + str(beta2)]
                        else:
                            _2hop_p = compute_2hop_p(beta1, beta2, T)
                            dic_2hop[str(beta1) + "_" + str(beta2)] = _2hop_p
                        if knode not in dic_path_prob:
                            dic_path_prob[knode] = np.array([1 - _2hop_p])
                        else:
                            dic_path_prob[knode] = np.append(dic_path_prob[knode], 1 - _2hop_p)

                        for lnode in neighbour_dict[knode]:
                            if lnode != jnode and lnode != inode:
                                beta3 = probability_mx[knode][lnode]
                                if str(beta1) + "_" + str(beta2) + "_" + str(beta3) in dic_3hop:
                                    _3hop_p = dic_3hop[str(beta1) + "_" + str(beta2) + "_" + str(beta3)]
                                else:
                                    _3hop_p = compute_3hop_p(beta1, beta2, beta3, T)
                                    dic_3hop[str(beta1) + "_" + str(beta2) + "_" + str(beta3)] = _3hop_p
                                if lnode not in dic_path_prob:
                                    dic_path_prob[lnode] = np.array([1 - _3hop_p])
                                else:
                                    dic_path_prob[lnode] = np.append(dic_path_prob[lnode], 1 - _3hop_p)

            p_path_fail = np.array([])
            for i in dic_path_prob:
                p_path_fail = np.append(p_path_fail, np.prod(dic_path_prob[i]))
            p_path_success = 1 - p_path_fail
            psum = np.append(psum, np.sum(p_path_success))
        _3hop_psum = psum + 1

    endtime = datetime.datetime.now()
    print("beta=", beta, "T=", T, "-------------- adaptive_3hopinf_NearExact time on :" + fileName, endtime - starttime)

    """compute influence(scale) of seeds"""
    scale_list = []
    for i in tqdm(range(1, len(seeds) + 1)):
        tmp_seeds = seeds[0:i]
        scale_list_temp = []
        for j in range(R):
            scale_temp, I_list = hs.hyperSI(df_hyper_matrix, tmp_seeds, beta, T)
            scale_list_temp.append(scale_temp)
        tmp_scale = sum(scale_list_temp) / len(scale_list_temp)
        scale_list.append(tmp_scale)
    return np.array(seeds), np.array(scale_list)

# K = 42
# R = 10
# T = 25
# beta = 0.01
# theta = 10000
# #
# fileName = 'Algebra'
# fileName = 'Restaurants-Rev'
# fileName = 'Music-Rev'
# fileName = 'Geometry'
# fileName = 'Bars-Rev'
# fileName = 'NDC-classes-unique-hyperedges'
# fileName = 'iJO1366'
# fileName = 'DAWN'       #mean degree = 90.90
# fileName = 'email-Enron'              #mean degree = 25.35

# datasets = ['Algebra', 'Music-Rev', 'NDC-classes-unique-hyperedges', 'email-Enron']
# df_hyper_matrix, N = tf.changeEdgeToMatrix('../data/' + fileName + '.txt')
# print(df_hyper_matrix)
# print(df_hyper_matrix.sum(axis=0))
# degree = getTotalAdj(df_hyper_matrix, N)
# print(sum(degree))
# print(sum(degree) / N)
# HyperDegree = df_hyper_matrix.sum(axis=1)
# Hdegree = np.array([])
# for item in HyperDegree:
#     Hdegree = np.append(Hdegree, item)
# print(HyperDegree)
# print(Hdegree)
# print(np.mean(Hdegree))
# hyperedgeSize = np.array([])
# for item in df_hyper_matrix.sum(axis=0):
#     hyperedgeSize = np.append(hyperedgeSize, item)
# # print(hyperedgeSize)
# print(np.mean(hyperedgeSize))

# data = get_inf_1hopinf_NearExact(fileName, df_hyper_matrix, N, beta, T)
# np.savetxt(fname="influence_1hopinf_NearExact/beta_5_T_1_theta_10000/" + fileName + ".txt", X=data, delimiter=',')

# _2hop_inf, probability_mx, neighbour_dict, dic = compute_2hop_inf(fileName, df_hyper_matrix, N, beta, T)
# print(_2hop_inf)

# seeds = [6]
# bi_adj, probability_mx, neighbour_dict, edge_set, edge_prob = Hyper_to_direct(df_hyper_matrix)
# dic_nbr_prob = {}
# for inode in range(N):
#     prob = np.array([])
#     for jnbr in neighbour_dict[inode]:
#         p = probability_mx[inode][jnbr]
#         prob = np.append(prob, p)
#     dic_nbr_prob[inode] = prob * beta
# scale_list = np.array([])
# cnt = 0
# for i in tqdm(range(theta)):
#     scale, I_list = DiSpreading_fast(probability_mx, neighbour_dict, dic_nbr_prob, df_hyper_matrix, seeds, beta, T)
#     scale_list = np.append(scale_list, scale)
#     for node in I_list:
#         if node == 6:
#             cnt += 11
# print(np.mean(scale_list))
# print(cnt/theta)

# _2hop_psum, probability_mx, neighbour_dict, dic1, dic2 = compute_2hop_inf_NearExact(fileName, df_hyper_matrix, N, beta, T)
# _3hop_psum, probability_mx, neighbour_dict, dic_1hop, dic_2hop, dic_3hop = compute_3hop_inf_NearExact(fileName, df_hyper_matrix, N, beta, T)
# print(_3hop_psum)

# seeds, scale_list = MIE_1(fileName, df_hyper_matrix, K, R, N, beta, T)
# for seed in seeds:
#     print(seed)
# print("-" * 100)
# for scale in scale_list:
#     print(scale)
# np.savetxt(fname="sensitivity/seeds_" + fileName + "_" + "_" + str(beta)+ "_" + str(T) + ".txt", X=seeds,
#            delimiter=',')
# np.savetxt(fname="sensitivity/scale_list_" + fileName + "_" + "_" + str(beta)+ "_" + str(T) + ".txt", X=scale_list,
#            delimiter=',')

# acc_list = MIE_2_test_acc(fileName, df_hyper_matrix, K, R, N, beta, T)
# acc_list = np.array(acc_list)
# np.savetxt(fname="acclist/MIE_2_" + fileName + "_" + "_" + str(beta)+ "_" + str(T) + ".txt", X=acc_list, delimiter=',')
# print(acc_list)

# seed = [1]
# p_sim, gt = simulate(df_hyper_matrix, N, theta, beta, T, seed)
# error = abs(gt - p_sim)
# print("theta=", theta, "\n")
# print(gt)
# print(p_sim)
# print(error)
