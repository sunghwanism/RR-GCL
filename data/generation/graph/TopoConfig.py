
SEED = 42

centrality_measures = {
    'betweenness': { # Shortest Path betweenness
        'k': None,  # If None, computes exact betweenness; otherwise, uses k random nodes as sources
        'normalized': True,
        'weight': None, # or name of edge attribute
        'seed': SEED
        },

    'edge_betweenness': { # Edge betweenness (https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html#networkx.algorithms.centrality.edge_betweenness_centrality)
        'k': None,  # If None, computes exact betweenness; otherwise, uses k random nodes as sources
        'normalized': False,
        'weight': None, # or name of edge attribute
        'seed': SEED
        }, 
    
    'flow_betweenness': {
        'normalized': True,
        'weight': None, # or name of edge attribute
        'solver': 'lu', # full - use most memory | lu - use less memory | cg - use least memory
        },
    
    'closeness': { # Harmonic Closeness
        'distance': None, # or name of edge attribute
        }, 
    
    'flow_closeness':{
        'weight':None,
        'solver': 'lu'# full - use most memory | lu - use less memory | cg - use least memory
        }, 
    'eigenvector': {
        'max_iter':1000,
        'weight': None # or name of edge attribute
        }, 
    
    'local_clustering':{
        'nodes': None, # or list of nodes to compute clustering for
        'weight': None # or name of edge attribute
        },
    
    'k_truss': {
        'k': 3 # k value for k-truss
         },
    'k_core': {
        'k': 28 # k value for k-core
        },
    'att_assortativity': {
        'att': ['has_mutation','total_mutations_count', 'unique_mutation_types_count', 'unique_patients_count']
    },
    'louvain_comm': {
        'weight': None, # or name of edge attribute || default 'weight' if you want to non-weighted use None
        'seed': SEED
    },
    'pagerank': {
        'alpha': 0.85,
        'max_iter': 500,
        'tol': 1.0e-6,
        'nstart': None,
        'weight': None, # or name of edge attribute (weight)
        'dangling': None
    },
    'avg_short_path_len': {
        'method': 'unweighted', # 'dijkstra' for weighted, 'unweighted' for unweighted
        'weight': None # or name of edge attribute
    },
    'MCC': {
        'normalize': True # whether to normalize the MCC values to [0,1]
    }
}