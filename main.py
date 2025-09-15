import os
import time

import hypernetx as hnx
import matplotlib
import numpy as np
import pandas as pd
import scienceplots
import scipy
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import trapz
from scipy.stats import pearsonr, entropy
from tqdm import tqdm

from IM.algo import get_vital_edge_community, compute_C, HACE_all, HCLI, greedy_prob, prob_update, HCLI_sparse, \
    HACE_all_sparse
from IM.models import influence_SI
from IM.my_algorithms import MIE_1, MIE_2, adeff, tf
from IM.topo.Hyper_topo import print_topo_info
from IM.utils import load_data, generate_H, compute_C_sparse, sparse_uint8_mult, \
    load_data_sim
from algorithms.Hyperspreading import Hyperspreading
from algorithms.algorithm import HACE, UseSeedSet, hurDisc, hs, HADP, HyperDegree
from algorithms.transform import Transform

print(plt.style.available)
matplotlib.use('module://backend_interagg')
matplotlib.use('PS')
plt.style.use(['science'])

# 刻度在内，设置刻度字体大小
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def plot_all(file, res_pd, columns, b, type):
    scales = []
    colors = ['r', 'g', 'y', 'k', 'm', 'tab:gray', 'cadetblue', 'sienna', 'orange']
    markers = ['o', 'v', 'x', '*', '+', 's', '.', 'P', '<', 'H', 'D', '>']
    with plt.style.context(['ieee']):
        for i in range(len(columns)):
            scale_temp = res_pd[columns[i]]
            scales.append(scale_temp)
            scale_temp = scale_temp.values.flatten()
            area = trapz(scale_temp, np.array(range(scale_temp.shape[0])), dx=0.001)
            print(columns[i], area)
            plt.plot(range(1, len(scale_temp) + 1), scale_temp)

        plt.xlabel('K')
        plt.ylabel(r'$\sigma$(S)')
        plt.xlim(1, None)
        plt.legend(columns)
        out_path = f'./fig/plot_large_scale/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        plt.savefig(f'{out_path}/{file}_{b}_{type}.eps', dpi=600)
        plt.savefig(f'{out_path}/{file}_{b}_{type}.png', dpi=600)
        plt.show()
    return scales


def run(fileName, R, b, t, s_df=None):
    tf = Transform()
    # df_hyper_matrix, N = tf.changeEdgeToMatrix('../data/' + fileName + '.txt')
    # draw(df_hyper_matrix, 'double', fileName)
    path = '../data/'
    if fileName in {'coauth-MAG-Geology', 'coauth-MAG-History', 'threads-stack-overflow'}:
        edges, N1 = load_data_sim(path, fileName)
    else:
        edges, N1 = load_data(path, fileName)
    H = hnx.Hypergraph({i: edges[i] for i in range(len(edges))})
    inc_matrix, node_inc_edge = generate_H(edges, N1)
    print("generate over!")
    K = int(0.1*inc_matrix.shape[0])
    start_time = time.time()
    # A = sparse_uint8_mult(inc_matrix, inc_matrix.T)
    # print("A over", time.time() - start_time)
    # 将 A 的对角元素设置为 0
    C, A = compute_C_sparse(inc_matrix)
    A.setdiag(0)
    rows, cols, data = scipy.sparse.find(C)
    C = {(i, j): val for i, j, val in zip(rows, cols, data)}
    # C = C.todense()
    print("C over", time.time()-start_time)
    # C = np.clip(C, 0, 1)
    # print("A over", time.time()-start_time)
    nodes = list(H.nodes())
    node_inc_node = {n: set(np.nonzero(A[n, :])[1]) for n in range(inc_matrix.shape[0])}
    print("node inc node over", time.time()-start_time)
    print("number of nodes:", len(nodes))
    print("average degree of nodes:", np.mean([len(node_inc_node[node]) for node in range(inc_matrix.shape[0])]))
    # HADP_scale_list = UseSeedSet(inc_matrix, [], K, R, seeds_HADP, node_inc_edge, edges, b, T)
    # HDegree_scale_list = UseSeedSet(inc_matrix, [], K, R, seeds_HDegree, node_inc_edge, edges, b, T)
    # print(seeds_HCLI, seeds_HACE, seeds_HADP, seeds_HDegree)
    # print(HACE_scale_list, HCLI_scale_list, HADP_scale_list, HDegree_scale_list)
    return

def get_single_node_influence(fileName, R, b, T):
    hs = Hyperspreading()
    tf = Transform()
    df_hyper_matrix, N = tf.changeEdgeToMatrix('../data/' + fileName + '.txt')
    # draw(df_hyper_matrix, 'double', fileName)
    path = '../data/'
    edges, N1 = load_data(path, fileName)
    inc_matrix, node_inc_edge = generate_H(edges, N1)
    H = hnx.Hypergraph({i: edges[i] for i in range(len(edges))})
    node_set = list(H.nodes())
    scale_list = []
    vital_edge, vital_edge_community = get_vital_edge_community(inc_matrix, 0.005)
    A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    for i in range(A.shape[0]):
        A[i, i] = 0
    node_inc_node = {n: set(np.nonzero(A[n, :])[1]) for n in list(H.nodes())}
    for i in tqdm(list(H.nodes())):
        scale_all = 0.
        for _ in range(R):
            scale, I_list = hs.hyperSI(df_hyper_matrix, [node_set.index(i)], b, T)
            scale_all += scale
        scale_all /= R
        node_inc_edge = set(np.nonzero(inc_matrix[i, :])[1])
        scale_list.append(
            (i, scale_all, len(node_inc_node[i]), len(node_inc_edge & set(vital_edge)) / len(node_inc_edge)))
    scale_list.sort(key=lambda x: x[1], reverse=True)

    # print(scale_list)
    seeds = [scale_list[i][0] for i in range(int(0.1 * len(node_set)))]
    seeds = [node_set.index(i) for i in seeds]
    return seeds


def corr_bet_inf_and_sigma_multi_nodes(df_hyper_matrix, inc_matrix, H, K, R, dataset, b, T):
    """
    GeneralGreedy algorithm
    """
    hs = Hyperspreading()
    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    nodes = list(H.nodes())
    C = compute_C(np.array(inc_matrix.todense()))
    C = np.clip(C, 0, 1)
    A = np.dot(inc_matrix, inc_matrix.T).astype(np.float64)
    for i in range(A.shape[0]):
        A[i, i] = 0
    node_inc_node = {n: set(np.nonzero(A[n, :])[1]) for n in nodes}
    seeds, scale_list = [], []
    last_scale = 0.
    pearson_ACC, pearson_CC, pearson_DCC, pearson_BCC = [], [], [], []
    pearson_g, pearson_h = [], []
    pearson_X = []
    contacted_prob = {n: 0. for n in nodes}
    N_s = set()
    X = [0] * inc_matrix.shape[0]
    for i in range(K):
        print(i)
        maxNode = 0
        maxScale = 1
        ACC, CC, sigma = [], [], []
        BCC = []
        g, h = [], []
        sigma_g, sigma_h = [], []
        DCC = []
        GIP = []
        seed_and_N_s = set(seeds) | set([nodes.index(n) for n in N_s])
        for inode in tqdm(range(0, len(degree))):
            infected_frequency = {num: 0 for num in range(inc_matrix.shape[0])}
            if inode in seeds:
                continue
            inf_N_s = 0.
            for n_N_s in N_s:
                P_N_s = contacted_prob[n_N_s]
                if P_N_s == 0:
                    inf_N_s += C[n_N_s, nodes[inode]]
                else:
                    inf_N_s = inf_N_s + (C[n_N_s, nodes[inode]] * C[n_N_s, nodes[inode]]) / (
                            C[n_N_s, nodes[inode]] + P_N_s)

            seeds.append(inode)
            scale_temp = 0.
            for _ in range(R):
                scale, I_list = hs.hyperSI(df_hyper_matrix, seeds, b, T)
                scale_temp += scale
            scale_temp = float(scale_temp / 10)
            if scale_temp > maxScale:
                maxNode = inode
                maxScale = scale_temp

            sigma_i = round(scale_temp - last_scale, 2)
            seeds.remove(inode)
            ACC_i = sum(C[:, nodes[inode]]) - sum([C[nodes[s], nodes[inode]] for s in seed_and_N_s]) + inf_N_s
            CC_i = sum(C[:, nodes[inode]]) - sum([C[nodes[s], nodes[inode]] for s in seeds])
            BCC_i = sum([C[nodes[inode], nodes[s]] for s in seeds])
            infed = sum([C[nodes[inode], nodes[s]] for s in seeds])
            if infed == 0:
                sigma_h.append(sigma_i)
                h.append(ACC_i)
            else:
                sigma_g.append(sigma_i)
                g.append(ACC_i / infed)
            # print(inode, inf1, CC_i)
            # infed = sum([C[nodes[inode], nodes[s]] for s in seeds])
            ACC.append(ACC_i)
            CC.append(CC_i)
            DCC.append((1 - infed * beta) * ACC_i)
            BCC.append(BCC_i)
            GIP.append(X[inode])
            sigma.append(sigma_i)
        seeds.append(maxNode)
        seed = maxNode
        X[maxNode] = 1
        inc_nodes = node_inc_node[seed] - set(seeds)
        N_s |= inc_nodes
        X_temp = [0] * len(X)
        for N_n in nodes:
            if N_n in seeds:
                X_temp[N_n] = 1
                continue
            if N_n in inc_nodes:
                contacted_prob[N_n] = C[N_n, seed] + contacted_prob[N_n]
            temp_not_infected = 1.
            for j in node_inc_node[N_n] - set(seeds):
                temp_not_infected = temp_not_infected * (1 - beta * C[N_n, j] * X[j])
            temp_not_infected = temp_not_infected * (1 - beta * C[N_n, seed])
            X_temp[N_n] = X[N_n] + (1 - temp_not_infected) * (1 - X[N_n])
            if X_temp[N_n] > 1:
                X_temp[N_n] = 1

        X = X_temp
        for N_n in node_inc_node[nodes[maxNode]]:
            contacted_prob[N_n] += C[N_n, nodes[maxNode]]
        N_s |= set(node_inc_node[nodes[maxNode]])
        last_scale = maxScale
        inf_spread_matrix.append(maxScale)
        if len(sigma) >= 2:
            r_ACC = round(pearsonr(ACC, sigma)[0], 2)
            r_CC = round(pearsonr(CC, sigma)[0], 2)
            r_DCC = round(pearsonr(DCC, sigma)[0], 2)
            r_BCC = round(pearsonr(BCC, sigma)[0], 2)
            r_X = round(pearsonr(GIP, sigma)[0], 2)
        else:
            r_ACC = 0.
            r_CC = 0.
            r_DCC = 0.
            r_BCC = 0.
            r_X = 0.
        if len(sigma_g) >= 2:
            r_g = round(pearsonr(g, sigma_g)[0], 2)
        else:
            r_g = 0.
        if len(sigma_h) >= 2:
            r_h = round(pearsonr(h, sigma_h)[0], 2)
        else:
            r_h = 0.
        pearson_ACC.append(r_ACC)
        pearson_CC.append(r_CC)
        pearson_DCC.append(r_DCC)
        pearson_BCC.append(r_BCC)
        pearson_X.append(r_X)
        pearson_g.append(r_g)
        pearson_h.append(r_h)
        print(r_ACC, r_BCC, r_X)

    with open("pearson_CC_ACC_1.txt", "w") as f:
        f.write(str(dataset))
        f.writelines(str(pearson_ACC))
        f.writelines(str(pearson_CC))
        f.writelines(str(pearson_DCC))
        f.writelines(str(pearson_g))
        f.writelines(str(pearson_h))
        f.writelines(str(pearson_X))
        f.writelines(str(pearson_BCC))

    # print("pearson_ACC:", pearson_ACC)
    # print("pearson_CC:", pearson_CC)
    # print("pearson_DCC:", pearson_DCC)
    print("pearson_BCC:", pearson_BCC)
    print("pearson_X:", pearson_X)
    # print("pearson_g:", pearson_g)
    # print("pearson_h", pearson_h)
    # plt.plot(range(K), pearson_ACC)
    # plt.savefig(f'fig/analyze/pearson_ACC_{dataset}.eps', dpi=600)
    # plt.show()
    #
    # plt.plot(range(K), pearson_CC)
    # plt.savefig(f'fig/analyze/pearson_CC_{dataset}.eps', dpi=600)
    # plt.show()
    #
    # plt.plot(range(K), pearson_DCC)
    # plt.savefig(f'fig/analyze/pearson_DCC_{dataset}.eps', dpi=600)
    # plt.show()

    plt.plot(range(K), pearson_BCC)
    plt.savefig(f'fig/analyze/pearson_BCC_{dataset}.eps', dpi=600)
    plt.show()

    plt.plot(range(K), pearson_X)
    plt.savefig(f'fig/analyze/pearson_X_{dataset}.eps', dpi=600)
    plt.show()

    # plt.plot(range(K), pearson_g)
    # plt.savefig(f'fig/analyze/pearson_g_{dataset}.eps', dpi=600)
    # plt.show()
    #
    # plt.plot(range(K), pearson_h)
    # plt.savefig(f'fig/analyze/pearson_h_{dataset}.eps', dpi=600)
    # plt.show()

    return inf_spread_matrix


def pearson_multi_seeds(b, t):
    for d in ['Algebra', 'Geometry', 'Restaurants-Rev']:
        tf = Transform()
        df_hyper_matrix, N = tf.changeEdgeToMatrix('../data/' + d + '.txt')
        # draw(df_hyper_matrix, 'double', fileName)
        path = '../data/'
        edges, N1 = load_data(path, d)
        H = hnx.Hypergraph({i: edges[i] for i in range(len(edges))})
        inc_matrix, node_inc_edge = generate_H(edges, N1)
        K = int(0.1 * len(list(H.nodes)))
        corr_bet_inf_and_sigma_multi_nodes(df_hyper_matrix, inc_matrix, H, K, 10, d, b, t)


def time_comparison(datasets_list, b, t):
    algos = ['HACE', 'HCLI', 'MIE(l=1)', 'MIE(l=2)', 'Adeff']
    R = 100
    for i, dataset in enumerate(datasets_list):
        # result_path = f'./7.11_{b}_{t}'
        # result = pd.read_csv(f"{result_path}/{dataset}.csv")
        times = pd.read_csv(f"times/{dataset}.csv")
        colors = ['r', 'g', 'y', 'k', 'm', 'tab:gray', 'cadetblue', 'sienna', 'orange']
        markers = ['o', 'v', 'x', '*', '+', 's', '.', 'P', '<', 'H', 'D', '>']
        x_max = 0
        for i in range(len(algos)):
            time_temp = times[algos[i]]
            x_max = len(time_temp)
            plt.plot(range(1, len(time_temp) + 1), time_temp)

        plt.xlabel('K')
        plt.ylabel(r'Time/s')
        plt.xlim(1, x_max)
        plt.ylim(0, None)
        plt.legend(algos)
        out_path = f'./fig/time_comparison/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        plt.savefig(f'{out_path}/{dataset}_{b}_time.eps', dpi=600)
        plt.savefig(f'{out_path}/{dataset}_{b}_time.png', dpi=600)
        plt.show()


def plot_different_para_comparison(datasets_list):
    results = pd.read_csv('different_para_comparison.csv')
    b_list = [0.005, 0.01, 0.015, 0.02]
    t_list = [10, 15, 20, 25]
    colors = ['r', 'g', 'tab:gray', 'cadetblue', 'sienna']
    markers = ['o', 'v', 'H', 'D', '>']
    labels = [r'$\beta^{H}=0.005$', r'$\beta^{H}=0.01$', r'$\beta^{H}=0.015$', r'$\beta^{H}=0.02$',
              'T=10', 'T=15', 'T=20', 'T=25']

    algos = ['HACE', 'HCLI', 'MIE(l=1)', 'MIE(l=2)', 'Adeff']
    for dataset in datasets_list:
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left,bottom,width,height
        ax2 = fig.add_axes([0.6, 0.25, 0.25, 0.25])  # left,bottom,width,height
        for i, algo in enumerate(algos):
            b_temp_list = []
            for b in b_list:
                b_temp_list.append(float(results[results['index'] == f'{dataset}_{b}_25'][algo].tolist()[0] / 1000))
            ax1.plot(b_list, b_temp_list)

        ax1.set_xlabel(r'$\beta^{H}$')
        ax1.set_ylabel('AUC')
        # plt.legend(algos)
        ax1.set_xticks(b_list)

        for i, algo in enumerate(algos):
            t_temp_list = []
            for t in t_list:
                t_temp_list.append(float(results[results['index'] == f'{dataset}_0.01_{t}'][algo].tolist()[0] / 1000))
            ax2.plot(t_list, t_temp_list)
        ax2.set_xlabel(r'$T$')
        ax2.set_ylabel('AUC')
        ax2.set_xticks(t_list)
        ax1.legend(algos, loc='upper left')
        out_path = f'./fig/para_comparison_thesis/'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        fig.savefig(f'{out_path}/{dataset}_com.eps', dpi=600)
        # fig.savefig(f'{out_path}/{dataset}_com.png', dpi=600)
        fig.show()

    return

def plot_large_scale():
    file_path = "large_scale.xlsx"  # 替换为你的文件路径
    sheet_names = ["DBLP", "cat-edge-MAG-10", "trivago", "coauth-MAG-History"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    # gs = GridSpec(3, 4, figure=fig)  # 3 行，4 列
    algos = ["HACE", "HCLI", "HADP", 'HDegree']
    colors = [
        '#D62728',  # 红色
        '#1F77B4',  # 鲜艳的蓝色
        '#2CA02C',  # 鲜艳的绿色
        '#FF7F0E',  # 鲜艳的橙色
        '#17BECF',  # 青色
        '#7F7F7F',  # 灰色
        '#8C564B',  # 棕色
        '#9467BD',  # 紫色
        '#BCBD22',  # 橄榄绿
    ]
    markers = ['o', 's', 'D', '^', 'v', '*', 'p', 'X', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    base_markersize = 15
    base_linewidth = 12
    # 假设所有子图的图例相同
    for k, ax in enumerate(axes.flat):
        result = pd.read_excel(file_path, sheet_name=sheet_names[k], usecols=algos)

        x_min, x_max = 1, len(result[algos[0]])
        y_min, y_max = min([min(result[algo]) for algo in algos]), max([max(result[algo]) for algo in algos])

        # 计算动态调整的比例因子
        x_range = x_max - x_min
        y_range = y_max - y_min
        scale_factor = np.sqrt(x_range ** 2 + y_range ** 2) / np.sqrt(10 ** 2 + 10 ** 2)  # 以 10x10 为基准

        # 动态调整 markersize 和 linewidth
        size = base_markersize / scale_factor
        width = base_linewidth / scale_factor
        for j, algo in enumerate(algos):
            ax.plot(range(1, len(result[algos[j]]) + 1), result[algos[j]], label=algos[j],
                    color=colors[j],
                    marker=markers[j],
                    linestyle=linestyles[j],
                    markersize=size,  # 设置 marker 大小
                    linewidth=width)
        ax.set_xlabel('K')
        ax.set_ylabel(r'$\sigma(\mathcal{S})$')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.text(0.5, -0.25, f'({chr(97 + k)})', transform=ax.transAxes,
                fontsize=12, ha='center', va='center')

    handles, labels = axes[0, 0].get_legend_handles_labels()

    # 在大图下方添加统一图例
    legend_col = 4
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=legend_col, fontsize='medium')

    # 调整子图间距
    fig.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1, hspace=0.4, wspace=0.2)

    # 保存图像
    # plt.savefig('output.png', dpi=300, bbox_inches='tight')

    fig.tight_layout()
    fig.savefig(f'large_scale_ppt.eps', dpi=600)
    fig.savefig(f'large_scale_ppt.png', dpi=600)
    fig.show()


if __name__ == '__main__':
    datasets = ['Algebra', 'Bars-Rev', 'Geometry', 'iAF1260b',
                'iJO1366', 'Music-Rev', 'NDC-classes-unique-hyperedges', 'Restaurants-Rev']
    # print_topo_info('../data/', 'cat-edge-MAG-10')
    beta, T = 0.01, 25
    b_l = [0.005, 0.01, 0.015, 0.02]
    t_l = [10, 15, 20, 25]
    # for dataset in datasets:
    #     run(dataset, 20, 0.01, 25)
    # run('Music-Rev', 50, 0.01, 25)
    # run('hyperedges-amazon-reviews', 10, 0.01, 25)
    # run('Cooking', 10, 0.01, 25)
    # run('DBLP', 20, 0.01, 25)
    # run('cat-edge-MAG-10', 20, 0.01, 25)
    # run('trivago', 20, 0.01, 25)
    # run('threads-stack-overflow', 20, 0.01, 25)
    # run('coauth-MAG-History', 20, 0.01, 25)
    # run('coauth-MAG-Geology', 20, 0.01, 25)
    # plot_large_scale()
    # plot_different_para_comparison(datasets)
    # plot_comparison_update_diff_para(datasets)
    # plot_comparison_update_time(datasets)
    # pearson_multi_seeds(beta, T)
    