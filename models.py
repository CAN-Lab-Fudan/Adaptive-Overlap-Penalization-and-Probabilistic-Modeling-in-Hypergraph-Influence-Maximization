import copy
import random

import numpy as np


def F_function(active_set, edge_i, s_function, threshold_low, threshold_high):
    """
    for an edge i
    """
    inter_set = set(active_set) & set(edge_i)
    if len(inter_set) == 0:
        return 0

    # g = float(len(inter_set) / len(edge_i))
    return s_function(len(inter_set) / len(edge_i), threshold_low, threshold_high)


def influence_opinion_process(seed_set, H, thresholds, s_function, threshold_low, threshold_high):
    """
    inspired by multi-body consensus model in hypergraph
    """
    active_nodes = copy.deepcopy(seed_set)
    new_active = copy.deepcopy(seed_set)
    while len(new_active) > 0:
        edges_of_new_nodes = set()
        for node in new_active:
            temp = list(np.nonzero(H[node])[1])
            edges_of_new_nodes |= set(temp)

        edges_of_new_nodes = list(edges_of_new_nodes)
        new_active = set()
        for edge in edges_of_new_nodes:
            node_in_edge = set(np.nonzero(H[:, edge])[0])
            temp_active = set()
            for node in list(node_in_edge - set(active_nodes) - new_active):
                edge_of_node = list(np.nonzero(H[node])[1])
                influence = max([F_function(active_nodes, list(np.nonzero(H[:, e])[0]),
                                            s_function, threshold_low, threshold_high) for e in edge_of_node])

                if influence > thresholds[node]:
                    # print(influence, thresholds[node])
                    temp_active.add(node)
            new_active |= temp_active
        print(new_active)
        active_nodes = list(set(active_nodes) | new_active)
    return len(active_nodes)


def influence_v3(seed_set, H, threshold):
    """
    have a similar process like models in epidemic.
    But influence probability is simple set by the ratio of active nodes in hyperedges.
    """
    active_nodes = copy.deepcopy(seed_set)
    new_active = copy.deepcopy(seed_set)

    for _ in range(20):
        edges_of_new_nodes = set()
        for node in new_active:
            temp = list(np.nonzero(H[node])[1])
            edges_of_new_nodes |= set(temp)

        edges_of_new_nodes = list(edges_of_new_nodes)
        new_active = []
        for edge in edges_of_new_nodes:
            nodes_of_edge = set(np.nonzero(H[:, edge])[0])
            active_node_in_edge = list(set(active_nodes) & nodes_of_edge)
            i_ratio = len(active_node_in_edge) / len(nodes_of_edge)
            if i_ratio <= threshold:
                inf_prob = i_ratio
            else:
                inf_prob = threshold

            for node in list(set(nodes_of_edge) - set(active_nodes)):
                if node in new_active:
                    continue

                # inf_prob = random.uniform(edge_weight_sum, 1.)
                judge = random.random()
                if judge < inf_prob:
                    # print(judge, inf_prob)
                    new_active.append(node)

        active_nodes = list(set(active_nodes) | set(new_active))
    return len(active_nodes)


def influence_SI(seed_set, node_inc_edge, edges, beta, T):
    active_nodes = set(seed_set)
    for i in range(0, T):
        infected_T = set()
        for node in active_nodes:
            adj_edges = list(node_inc_edge[node])
            if len(adj_edges) > 0:
                edge = random.choice(adj_edges)
            else:
                continue
            adj_nodes = edges[edge]
            random_list = np.random.random(size=len(adj_nodes))
            infected_T |= set([adj_nodes[k] for k in np.where(random_list <= beta)[0]])

        active_nodes |= infected_T

    return len(active_nodes), active_nodes


def influence_edge_node_t(seed_set, H, weights, node_t, edge_t):
    active_nodes = copy.deepcopy(seed_set)
    node_num, edge_num = H.shape
    # init active edges
    active_edges = []
    for edge in range(edge_num):
        edge_nodes = set(np.nonzero(H[:, edge])[0])
        if len(edge_nodes) == 0:
            continue
        if len(set(active_nodes) & edge_nodes) / len(edge_nodes) > edge_t[edge]:
            active_edges.append(edge)

    # influence
    new_active_e = copy.deepcopy(active_edges)
    while len(new_active_e) != 0:
        new_active_n = []
        adj_nodes = set()
        for edge in new_active_e:
            adj_nodes |= set(np.nonzero(H[:, edge])[0])

        for node in set(adj_nodes-set(active_nodes)):
            adj_edges = set(np.nonzero(H[node, :])[1])
            # inf = sum([weights[node, e] for e in (set(active_edges) & adj_edges)])
            inf = len(set(active_edges) & adj_edges) / len(adj_edges)
            if inf > node_t[node]:
                new_active_n.append(node)

        if len(new_active_n) == 0:
            break

        active_nodes += new_active_n
        new_active_e = []
        for edge in (set(range(edge_num))-set(active_edges)):
            edge_nodes = set(np.nonzero(H[:, edge])[0])
            if len(set(active_nodes) & edge_nodes) / len(edge_nodes) > edge_t[edge]:
                new_active_e.append(edge)

        active_edges += new_active_e

    return len(active_nodes)


def influence_IC(seed_set, H, edge_t, node_t, p_node_to_edge, p_edge_to_node):
    """
    p_edge_to_node: {edge: {node: p}}
    p_node_to_edge: {node: {edge: p}}
    """
    active_nodes = copy.deepcopy(seed_set)
    last_active_n = copy.deepcopy(active_nodes)
    last_active_e = []
    active_edges = []
    edge_active_nodes = {i: [] for i in range(H.shape[1])}
    node_active_edges = {i: [] for i in range(H.shape[0])}
    while len(last_active_n) != 0 or len(last_active_e) != 0:
        new_active_n = []
        new_active_e = []
        adj_edge_all = set()
        for node in last_active_n:
            adj_edge = set(np.nonzero(H[node, :])[1]) - set(active_edges)
            adj_edge_all |= adj_edge
            # print("adj edges:", adj_edge)

            for edge in adj_edge:
                if random.random() < p_node_to_edge[node][edge]:
                    edge_active_nodes[edge].append(node)

        for edge in adj_edge_all:
            if len(edge_active_nodes[edge]) / len(set(np.nonzero(H[:, edge])[0])) > edge_t[edge]:
                # print(edge_active_nodes[edge], set(np.nonzero(H[:, edge])[0]), edge_t[edge])
                new_active_e.append(edge)
                active_edges.append(edge)

        # print(new_active_e)

        for edge in new_active_e:
            node_in_edge = set(np.nonzero(H[:, edge])[0])
            not_need = set(active_nodes) | set(new_active_n)
            alt_nodes = node_in_edge - not_need
            for n in alt_nodes:
                if random.random() < p_edge_to_node[edge][n]:
                    new_active_n.append(n)
                    active_nodes.append(n)

        # print("last active nodes:", last_active_n)
        # print("active edges:", active_edges)

        last_active_n = new_active_n
        last_active_e = new_active_e

    # print(active_nodes)
    return len(active_nodes)
