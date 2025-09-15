
import numpy as np
import pandas as pd
import random
import copy
import Hyperspreading
import networkx as nx
from tqdm import tqdm
import time
from Fitness import fitness

class algorithms:
   
    # 1、方法一：基于HEDV的greedy搜索方法
    def obj_func_greedy(self, df_hyper_matrix, k, obj_func_name):
        """
        基于目标函数的贪婪策略构建初始解
        """
        begin_time = time.time()
        seed_list_HEDV = []
        obj_func = algorithms.select_obj_func(self, obj_func_name)
        num_nodes = df_hyper_matrix.shape[0]
        seeds_Greedy = []
        for i in tqdm(range(k),desc='HEDV-greedy'): # 一共要添加k个节点
            maxNode = 0
            maxfitness = 0
            for inode in range(num_nodes):
                if inode not in seeds_Greedy:
                    seeds_Greedy.append(inode)
                    fitness = obj_func(self, df_hyper_matrix.values, seeds_Greedy)
                    seeds_Greedy.remove(inode)
                    if fitness > maxfitness:
                        maxNode = inode
                        maxfitness = fitness
            seeds_Greedy.append(maxNode)
            seed_list_HEDV.append(seeds_Greedy.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HEDV, cost_time
    # 辅助函数
    # 1、选择目标函数的计算方法
    def select_obj_func(self, obj_func_name):
        if obj_func_name == 'HEDV':
            return fitness.HEDV

