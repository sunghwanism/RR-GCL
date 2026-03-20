
import os
import sys

import networkx as nx
from joblib import Parallel, delayed, parallel_backend
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat

from pprint import pprint

import math
import numpy as np
import pandas as pd

from utils import save_graph

from datetime import datetime
ts = datetime.now().strftime("%Y%m%d")

_GLOBAL_G = None
def _init_worker(G):
    global _GLOBAL_G
    _GLOBAL_G = G

def spl_worker(node):
    lengths = nx.single_source_shortest_path_length(_GLOBAL_G, node)
    avg = sum(lengths.values()) / (len(_GLOBAL_G)-1) if len(_GLOBAL_G) > 1 else 0
    return node, avg

def chunk_nodes(nodes, n_jobs):
    nodes = list(nodes)
    total = len(nodes)
    chunk_size = math.ceil(total / n_jobs)
    for i in range(0, total, chunk_size):
        yield nodes[i:i + chunk_size]


def degree_centrality(G, device):
    deg = nx.degree_centrality(G)
    nx.set_node_attributes(G, deg, 'degree')
    return G

def get_transivity(G, device):
    result = round(nx.transitivity(G), 6) 
    print(f"[Transivity]: {result}")
    return result

def get_triangles(G, device):
    result = nx.triangles(G)
    nx.set_node_attributes(G, result, 'triangles')
    return result

def get_k_truss(G, device, param):
    result = {}
    max_k = 100
    for i in range(3, max_k):
        S = nx.k_truss(G, k=i)
        if len(S.nodes) == 0:
            break
        else:
            result[i] = len(S.nodes)
    try:
        print("Max k-truss:", max(result.keys()))
    except:
        print("No k-truss subgraph found.")
        result ={-1: -1}
    
    return result

def get_k_core(G):
    core_num = nx.core_number(G)
    nx.set_node_attributes(G, core_num, f'k_core')
    max_k = max(core_num.values())
    print("Max k-core:", max_k)
    
    return G, max_k

def get_betweenness(G, device, param, n_jobs=-1):
    if device == 'cpu':
        print(f"Using {n_jobs} CPU cores")

        with parallel_backend("loky", n_jobs=n_jobs):
            with nx.config.backends.parallel(
                n_jobs=n_jobs,
                backend="loky",
                inner_max_num_threads=1,
            ):
                btw = nx.betweenness_centrality(
                    G,
                    backend="parallel",
                    get_chunks=lambda nodes: chunk_nodes(nodes, n_jobs),
                    **param
                )
        nx.set_node_attributes(G, btw, 'betweenness')
        
    else:
        btw = nx.betweenness_centrality(G, backend="cugraph", **param)
        nx.set_node_attributes(G, btw, 'betweenness')
        
    return G

def get_closeness(G, params):
    clo = nx.harmonic_centrality(G, distance=None)#, **params)
    # clo = nx.closeness_centrality(G, **params)
    nx.set_node_attributes(G, clo, 'closeness_centrality')
    return G

def get_flow_closeness(G, params):
    flow_clo = nx.current_flow_closeness_centrality(G, **params)
    nx.set_node_attributes(G, flow_clo, 'flow_closeness')
    return G

def get_eign_vector(G, params, dummy_value=-1):
    try:
        eig = nx.eigenvector_centrality(G, **params)
        
    except Exception as e:
        eig = {node: dummy_value for node in G.nodes()}
    
    nx.set_node_attributes(G, eig, "eigenvector")
    return G

def get_local_clustering(G, params):
    clustering = nx.clustering(G, **params)
    nx.set_node_attributes(G, clustering, 'local_clustering')
    return G

def get_articulation_points(G):
    artp = list(nx.articulation_points(G))
        
    nx.set_node_attributes(
        G,
        {node: (1 if node in artp else 0) for node in G.nodes()},
        'articulation_point'
    )
    return G

def calc_num_articulation_points(G, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda d: 1 if d.get("articulation_point", 0) == 1 else 0)(d)
        for _, d in G.nodes(data=True)
    )
    return sum(results)

def get_avg_short_path_len(G,device, param):
    if device == 'cpu':
        if len((list(nx.connected_components(G))))==1:
            result = nx.average_shortest_path_length(G, **param)
            
        elif len((list(nx.connected_components(G))))>1:
            print("The graph is not connected. Calculating average shortest path length for each connected component...")
            result = []
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                n = len(subgraph)
                avg_len = nx.average_shortest_path_length(subgraph, **param)
                result.append((avg_len, n))

    return result
    
def get_biconnected_compt(G): # To-do
    '''Returns a generator of sets of nodes, one set for each biconnected component of the graph G.'''
    if not nx.is_biconnected(G):
        print("The Graph is not biconnected")
        
    print("Generate biconnected components set...")
    bicomp = list(nx.biconnected_components(G))
    
    return bicomp

def get_deg_assortativity(G):
    result = round(nx.degree_assortativity_coefficient(G), 5)
    return result

def get_alg_connectivity(G):
    result = nx.algebraic_connectivity(G, weight=None, method='lobpcg')
    return result

def get_bridges(G):
    bridges = list(nx.bridges(G))
    result = len(bridges)
    nx.set_edge_attributes(
        G,
        {(u, v): 1 for u, v in bridges},
        'bridge'
    )
    return G, result

def get_biconnected_components(G):
    '''Returns a generator of sets of nodes, one set for each biconnected component of the graph G.'''
    if not nx.is_biconnected(G):
        print("The Graph is not biconnected")
        
    print("Generate biconnected components set...")
    bicomp = list(nx.biconnected_components(G))
    
    return bicomp


def get_SpectralRadius(G, tolerance=1e-6, max_iterations=10000):
    for u, v, d in G.edges(data=True):
        d['weight'] = abs(d['weight'])
    
    if G.number_of_nodes() == 0:
        print("[Warning] The graph has no nodes.")
        return -1.0

    n = G.number_of_nodes()

    use_sparse = True
    try:
        A = nx.adjacency_matrix(G, weight='weight').astype(float)  # scipy.sparse matrix
    except Exception:
        A = nx.to_numpy_array(G, weight='weight', dtype=float)     # numpy dense
        use_sparse = False

    if not G.is_directed():
        try:
            from scipy.sparse.linalg import eigsh
            if use_sparse:
                vals = eigsh(A, k=1, which='LA', return_eigenvectors=False,
                             tol=tolerance, maxiter=max_iterations)
                return float(vals[0])
            else:
                w = np.linalg.eigvalsh(A)
                return float(np.max(w))
        except Exception:
            pass

    v = np.ones(n, dtype=float)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return 0.0
    v = v / v_norm

    prev_lambda = 0.0

    for _ in range(max_iterations):
        Av = A.dot(v) if use_sparse else (A @ v)
        norm_Av = np.linalg.norm(Av)
        if norm_Av == 0.0:
            return 0.0

        lambda_est = float(np.vdot(v, Av).real)

        v_new = Av / norm_Av

        if abs(lambda_est - prev_lambda) < tolerance or np.linalg.norm(v_new - v) < np.sqrt(tolerance):
            return abs(lambda_est)

        v = v_new
        prev_lambda = lambda_est

    print("Warning: Maximum iterations reached without convergence.")
    return abs(prev_lambda)

def get_louvain_comm(G, params):
    if params['weight'] is None:
        prefix = 'Nonweight'
    else:
        prefix = 'weight'
    H = G.copy()
    # for u, v, d in H.edges(data=True):
    #     d['weight'] = abs(d['weight'])
    community = nx.community.louvain_communities(H, **params)
    
    for com in community:
        for node in com:
            G.nodes[node][f'{prefix}_louvain_comm'] = community.index(com)
            
    return G

def get_shortest_path_length_per_node(G, n_jobs=-1):
    nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    # chunksize = 1
    chunksize = max(1, len(nodes) // (n_jobs * 8) or 1)

    m = {}
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(G,)) as ex:
        for node, avg in ex.map(spl_worker, nodes, chunksize=chunksize):
            m[node] = avg

    nx.set_node_attributes(G, m, "shortest_path_length")
    return G

def get_global_efficiency(G):
    efficiency = nx.global_efficiency(G)
    return efficiency

def get_local_efficiency(G):
    n_jobs = os.cpu_count() if os.cpu_count() is not None else -1
    
    with parallel_backend("loky", n_jobs=n_jobs):
        with nx.config.backends.parallel(
            n_jobs=n_jobs,                   # use all CPUs
            backend="loky",              # joblib default process backend
            inner_max_num_threads=1,  # avoid oversubscription from BLAS, etc.
        ):
            local_efficiency = nx.local_efficiency(G)
    return local_efficiency
    
def get_edge_betweenness_centrality(G, params=None, n_jobs=-1):

    if params is None:
        params = {}

    with parallel_backend("loky", n_jobs=n_jobs):
        with nx.config.backends.parallel(
            n_jobs=n_jobs,
            backend="loky",
            inner_max_num_threads=1,
        ):
            edge_bet = nx.edge_betweenness_centrality(
                G,
                backend="parallel",
                get_chunks=lambda nodes: chunk_nodes(nodes, n_jobs),
                **params
            )

    nx.set_edge_attributes(G, edge_bet, 'edge_betweenness')
    return G

def get_flow_betweenness_centrality(G, params):
    flow_bet = nx.current_flow_betweenness_centrality(G, **params)
    nx.set_node_attributes(G, flow_bet, 'flow_betweenness')
    return G

def get_att_assortativity(G, att, numeric=False):
    if numeric:
        r = nx.numeric_assortativity_coefficient(G, att)
    else:
        r = nx.attribute_assortativity_coefficient(G, att)
    
    return r

def get_densest_subgraph(G):
    density, subset_nodes = nx.algorithms.approximation.densest_subgraph(G, iterations=1, method='greedy++')    
    return density, subset_nodes

# def get_weighted_shortest_path_length(G):
#     H = G.copy()
        
#     for u, v, d in H.edges(data=True):
#         d['weight'] = abs(d['weight'])
        
#     H = nxp.ParallelGraph(H)
#     lengths = dict(nxp.all_pairs_dijkstra_path_length(H, weight='weight'))

#     node_attr = {}
#     for src, dist_dict in lengths.items():
#         if len(dist_dict) > 1:  # 자기 자신만 있는 경우 방지
#             avg_dist = sum(dist_dict.values()) / (len(dist_dict) - 1)
#         else:
#             avg_dist = 0
#         node_attr[src] = -avg_dist
        
#     nx.set_node_attributes(G, node_attr, "avg_w_shortest_path_length")
#     return G

def mcc_worker(G, node, clique_list):
    nbrs = list(G.neighbors(node))
    if len(nbrs) <= 1:
        return node, G.degree(node)
    
    if G.subgraph(nbrs).number_of_edges() == 0:
        return node, G.degree(node)

    total = 0
    for clique in clique_list:
        if node in clique:
            total += math.factorial(len(clique) - 1)
    return node, total

def get_MCC(G, params):
    n_jobs = os.cpu_count() if os.cpu_count() is not None else -1
    
    nodes = list(G.nodes())
    clique_list = list(nx.find_cliques(G))
    clique_list = [set(c) for c in clique_list if len(c) > 1]
    len_cliq_list = [len(c) for c in clique_list]
    max_cliq = max(len_cliq_list)
    
    mcc = {}
    chunksize = max(1, len(nodes) // (n_jobs * 8) or 1)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for node, score in ex.map(mcc_worker, 
                                  repeat(G),
                                  nodes, 
                                  repeat(clique_list),
                                  chunksize=chunksize):
            mcc[node] = score
            
    if params.get('normalize', True):
        for n in nodes:
            mcc[n] /= math.factorial(max_cliq - 1)
    nx.set_node_attributes(G, mcc, 'mcc')
    
    return G

def get_pagerank(G, params):
    pagerank = nx.pagerank(G, **params)
    nx.set_node_attributes(G, pagerank, 'pagerank')
    return G

def calcuator(G, config, device, centrality_measures, index=None, n_jobs=-1):
    if index is None:
        index = ''
    else:
        index = "_"+str(index)
    
    measures = config.measure
    if not isinstance(measures, list):
        measures = [measures]

    result_table_path = os.path.join(config.savepath, f'result.csv')
    
    try:
        result_table = pd.read_csv(result_table_path)
    except FileNotFoundError:
        result_table = pd.DataFrame({'Measure': [], 'Value': []})

    new_results = []
    
    copyG = G.copy()
    selfLoop = nx.selfloop_edges(copyG)
    
    if len(list(selfLoop)) > 0:
        copyG.remove_edges_from(selfLoop)
        print(f"[Info] Number of self-loop edges removed: {len(list(nx.selfloop_edges(G)))}")
    
    for measure in measures:
        print(f"Calculating {measure}...")
        if measure == 'degree':
            G = degree_centrality(copyG, device)
            save_graph(G, config.savepath, measure)
        
        elif measure == 'transitivity':
            result = get_transivity(copyG, device)
            new_results.append({'Measure': measure + f"{index}", 'Value': result})
            
        elif measure == 'triangles':
            result = get_triangles(copyG, device)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'k_truss':
            param = centrality_measures[measure]
            result = get_k_truss(copyG, device, param)
            new_results.append({'Measure': f"{measure}" + f"{index}", 'Value': [result]})
            
        elif measure == 'k_core':
            param = centrality_measures[measure]
            G, max_k = get_k_core(copyG)
            save_graph(G, config.savepath, measure)
            new_results.append({"Measure": 'k-core_kmax', "Value": max_k})
            
        elif measure == 'betweenness':
            param = centrality_measures[measure]
            G = get_betweenness(copyG, device, param, n_jobs=n_jobs)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'closeness':
            param = centrality_measures[measure]
            G = get_closeness(copyG, param)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'flow_closeness':
            param = centrality_measures[measure]
            G = get_flow_closeness(copyG, param)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'eigenvector':
            param = centrality_measures[measure]
            G = get_eign_vector(copyG, param)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'local_clustering':
            param = centrality_measures[measure]
            G = get_local_clustering(copyG, param)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'articulation_point':
            G = get_articulation_points(copyG)
            num_articulation_points = calc_num_articulation_points(copyG, n_jobs=n_jobs)
            print(f"[Number of Articulation Points]: {num_articulation_points}")
            new_results.append({'Measure': 'articulation_points_num' + f"{index}", 'Value': num_articulation_points})
            save_graph(G, config.savepath, measure)
            
        elif measure == 'avg_short_path_len':
            param = centrality_measures[measure]
            result = get_avg_short_path_len(copyG, device, param)
            new_results.append({'Measure': measure + f"{index}", 'Value': result})
            
        elif measure == 'deg_assortativity':
            result = get_deg_assortativity(copyG)
            new_results.append({'Measure': measure + f"{index}", 'Value': result})
        
        elif measure == 'alg_connectivity':
            result = get_alg_connectivity(copyG)
            new_results.append({'Measure': measure + f"{index}", 'Value': result})

        elif measure == 'bridges':
            G, result = get_bridges(copyG)
            num_bridges = result
            print(f"[Number of Bridges]: {num_bridges}")
            new_results.append({'Measure': 'bridges_num' + f"{index}", 'Value': num_bridges})
            save_graph(G, config.savepath, measure)
            
        elif measure == 'biconnected_components':
            bicomp = get_biconnected_components(copyG)
            num_bicomp = len(bicomp)
            print(f"[Number of Biconnected Components]: {num_bicomp}")
            new_results.append({'Measure': 'biconnected_components' + f"{index}", 'Value': [bicomp]})
            
        elif measure == 'spectral_radius':
            spectral_radius = get_SpectralRadius(copyG)
            print(f"[Spectral Radius]: {spectral_radius}")
            new_results.append({'Measure': 'spectral_radius' + f"{index}", 'Value': spectral_radius})

        elif measure == 'flow_betweenness':
            param = centrality_measures[measure]
            G = get_flow_betweenness_centrality(copyG, param)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'edge_betweenness':
            param = centrality_measures[measure]
            G = get_edge_betweenness_centrality(copyG, param, n_jobs)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'att_assortativity':
            att_list = centrality_measures[measure]['att']
            for att in att_list:
                numeric = att != 'has_mutation'
                r = get_att_assortativity(copyG, att, numeric)
                print(f"[Attribute Assortativity ({att})]: {r}")
                new_results.append({'Measure': f"att_assortativity_{att}" + f"{index}", 'Value': r})
        
        elif measure == 'shortest_path_length_per_node':
            G = get_shortest_path_length_per_node(copyG, n_jobs)
            save_graph(G, config.savepath, measure)
            
        elif measure == 'global_efficiency':
            result = get_global_efficiency(copyG)
            print(f"[Global Efficiency]: {result}")
            new_results.append({'Measure': measure + f"{index}", 'Value': result})
            
        elif measure == 'local_efficiency':
            result = get_local_efficiency(copyG)
            print(f"[Local Efficiency]: {result}")
            new_results.append({'Measure': measure + f"{index}", 'Value': result})
            
        elif measure == 'w_shortest_path_length':
            G = get_weighted_shortest_path_length(copyG)
            save_graph(G, config.savepath, measure)
        
        elif measure == 'louvain_comm':
            G = get_louvain_comm(copyG, centrality_measures[measure])
            save_prefix = 'Nonweight' if centrality_measures[measure]['weight'] is None else 'weight'
            save_graph(G, config.savepath, f"{save_prefix}_louvain_comm")
            
        elif measure == 'densest_subgraph':
            density, subset_nodes = get_densest_subgraph(copyG)
            print(f"[Densest Subgraph Density]: {density}, [Number of Nodes]: {len(subset_nodes)}")
            new_results.append({'Measure': 'densest_subgraph_density' + f"{index}", 'Value': density})
            new_results.append({'Measure': 'densest_subgraph_nodes' + f"{index}", 'Value': [subset_nodes]})
        
        elif measure == 'MCC':
            G = get_MCC(copyG, centrality_measures[measure])
            save_graph(G, config.savepath, measure)
            
        elif measure == 'pagerank':
            param = centrality_measures[measure]
            G = get_pagerank(copyG, param)
            save_graph(G, config.savepath, measure)
        
        else:
            print(f"[Error] The measure '{measure}' is not supported.")
            return 'Error'

    if new_results:
        new_results_df = pd.DataFrame(new_results)
        result_table = pd.concat([result_table, new_results_df]).drop_duplicates(subset=['Measure'], keep='last')
        result_table.to_csv(result_table_path, index=False)
    
    return "Success"