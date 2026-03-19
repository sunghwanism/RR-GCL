import os
import pickle

import networkx as nx



def load_graph(path: str) -> nx.Graph:
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

def save_graph(G: nx.Graph, path: str):
    with open(path, 'wb') as f:
        pickle.dump(G, f)

def get_node_att_value(obj, att):
    """Helper to get attribute value from a Graph node or dict."""
    if obj is None:
        raise ValueError("Input object is None.")
    if isinstance(obj, nx.Graph):
        return [d.get(att, None) for n, d in obj.nodes(data=True)]
    elif isinstance(obj, dict):
        return obj.get(att, None)
    else:
        raise TypeError("Input must be a networkx Graph or a node attribute dictionary.")

def get_edge_att_value(G, att):
    """Helper to get attribute value from Graph edges."""
    return [d.get(att, None) for u, v, d in G.edges(data=True)]

def get_sample(G):
    """Returns a sample node and edge from the graph."""
    sample_node = next(iter(G.nodes))
    sample_edge = next(iter(G.edges))
    return sample_node, sample_edge