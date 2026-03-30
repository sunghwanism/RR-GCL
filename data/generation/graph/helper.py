import os

# from CDSLoader import calculate_mutability_for_row
import networkx as nx

def check_path(EDGEPATH, INTERMEDIATEPATH, FINALPATH, REFPATH):
    if not os.path.exists(EDGEPATH):
        raise FileNotFoundError(f"Edge path not found: {EDGEPATH}")
        
    if not os.path.exists(REFPATH):
        raise FileNotFoundError(f"Reference path not found: {REFPATH}")

    if not os.path.exists(INTERMEDIATEPATH):
        os.makedirs(INTERMEDIATEPATH)

    if not os.path.exists(FINALPATH):
        os.makedirs(FINALPATH)


def load_pdb_uniprot_node(args):
    if args.EXCEPT_PDB: 
        except_pdb = []
        for file in args.EXCEPT_PDB: 
            except_pdb_in_df = pd.read_csv(os.path.join(args.REFPATH, file))
            except_pdb.extend(except_pdb_in_df[except_pdb_in_df.columns[0]].tolist())
        except_pdb = list(set(except_pdb))
        print(f"Unique PDBs to exclude: {len(except_pdb)}")
        except_pdb = [pdb.upper() for pdb in except_pdb]
    else:
        except_pdb = []

    if args.INCL_PDB:
        incl_pdb = []
        for file in args.INCL_PDB:
            incl_pdb_in_df = pd.read_csv(os.path.join(args.REFPATH, file))
            incl_pdb.extend(incl_pdb_in_df[incl_pdb_in_df.columns[0]].tolist())
        incl_pdb = list(set(incl_pdb))
        print(f"Unique PDBs to include: {len(incl_pdb)}")
        incl_pdb = [pdb.upper() for pdb in incl_pdb]
    else:
        incl_pdb = []

    if args.INCL_UNIPROT:
        incl_uniprot = []
        for file in args.INCL_UNIPROT:
            incl_uniprot_in_df = pd.read_csv(os.path.join(args.REFPATH, file))
            incl_uniprot.extend(incl_uniprot_in_df[incl_uniprot_in_df.columns[0]].tolist())
        incl_uniprot = list(set(incl_uniprot))
        print(f"Unique UniProts to include: {len(incl_uniprot)}")
        incl_uniprot = [uniprot.lower() for uniprot in incl_uniprot]
    else:
        incl_uniprot = []
    
    if args.EXCEPT_UNIPROT:
        except_uniprot = []
        for file in args.EXCEPT_UNIPROT:
            except_uniprot_in_df = pd.read_csv(os.path.join(args.REFPATH, file))
            except_uniprot.extend(except_uniprot_in_df[except_uniprot_in_df.columns[0]].tolist())
        except_uniprot = list(set(except_uniprot))
        print(f"Unique UniProts to exclude: {len(except_uniprot)}")
        except_uniprot = [uniprot.lower() for uniprot in except_uniprot]
    else:
        except_uniprot = []

    if args.INCL_NODE:
        incl_node = []
        for file in args.INCL_NODE:
            incl_node_in_df = pd.read_csv(os.path.join(args.REFPATH, file))
            incl_node.extend(incl_node_in_df['nodeid'].tolist())
        incl_node = list(set(incl_node))
        print(f"Unique Nodes to include: {len(incl_node)}")
    else:
        incl_node = []

    if args.EXCEPT_NODE:
        except_node = []
        for file in args.EXCEPT_NODE:
            except_node_in_df = pd.read_csv(os.path.join(args.REFPATH, file))
            except_node.extend(except_node_in_df['nodeid'].tolist())
        except_node = list(set(except_node))
        print(f"Unique Nodes to exclude: {len(except_node)}")
    else:
        except_node = []
    
    return incl_pdb, incl_uniprot, incl_node, except_pdb, except_uniprot, except_node

####################################
# Preprocessing Function
####################################

def generate_nodeid_and_only_uniprot(edge_df):
    edge_df['nodeid_1'] = (edge_df['uniprot1'].str.replace('_', '-', regex=False).astype(str) + "_" +
    edge_df['uniprot_resi1'].astype(str) + "_" +
    edge_df['res3n1'].astype(str)).str.lower()

    edge_df['nodeid_2'] = (edge_df['uniprot2'].str.replace('_', '-', regex=False).astype(str) + "_" +
    edge_df['uniprot_resi2'].astype(str) + "_" +
    edge_df['res3n2'].astype(str)).str.lower()

    edge_df['remove_homo_uniprot1'] = edge_df['uniprot1'].str.split("_").str[0]
    edge_df['remove_homo_uniprot2'] = edge_df['uniprot2'].str.split("_").str[0]

    return edge_df

def remove_ubq_related_connection(edge_df, remove_ubq_list):
    past_len = len(edge_df)
    edge_df = edge_df[~((edge_df['remove_homo_uniprot1'].isin(remove_ubq_list)) | (edge_df['remove_homo_uniprot2'].isin(remove_ubq_list)))]

    print(f"Remove ubq related Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")
    
    return edge_df

def filter_only_nucleosome_related_connection(edge_df, excl_pdb_df, incl_uniprot_df):
    past_len = len(edge_df)

    excl_pdb_list = excl_pdb_df['0'].str.lower().unique().tolist()
    edge_df = edge_df[~edge_df['pdb_code'].isin(excl_pdb_list)].reset_index(drop=True)
    
    print(f"Remove excl pdb Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")

    past_len = len(edge_df)
    incl_list = incl_uniprot_df['uniprot'].str.lower().unique().tolist()

    mask = (edge_df['remove_homo_uniprot1'].isin(incl_list)) & \
           (edge_df['remove_homo_uniprot2'].isin(incl_list))
    
    edge_df = edge_df[mask].reset_index(drop=True)

    print(f"Filter only nucleosome related Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")
    
    return edge_df

def remove_negative_and_zero_position_node(edge_df):
    past_len = len(edge_df)
    edge_df = edge_df[(edge_df['pdb_auth_resi1'].astype(int) > 0) & (edge_df['pdb_auth_resi2'].astype(int) > 0)].reset_index(drop=True)
    print(f"Remove negative and zero position node Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")
    return edge_df


def ready_for_CDS_filter(args, CDSTable, cancer_type='pan-cancer'):
    with open(os.path.join(args.REFPATH, args.MUTABILITYTABLE), 'r') as f:
        mutability_dict = json.load(f)
    if cancer_type == 'pan-cancer':
        mutability_dict = mutability_dict['33999']
    
    else:
        raise ValueError(f"Cancer type {cancer_type} is not supported.")

    CDSTable['mutability'] = CDSTable.apply(lambda row: calculate_mutability_for_row(row, mutability_dict['mutation_freq']), axis=1)
    CDSTable['mut_count'] = CDSTable.apply(lambda row: calculate_mutability_for_row(row, mutability_dict['mutartion_count']), axis=1)

    CDSTable = CDSTable[['node_id', 'unique_cds_contexts', 'mutability', 'mut_count']]
    CDSTable.rename({'mut_count':'mutagene_mut_count'}, inplace=True, axis=1)

    CDSTable = CDSTable[CDSTable['mutability']!=0]
    CDSTable.drop_duplicates('node_id', keep='first', inplace=True)
    
    return CDSTable


def merge_energy_nodes(df):

    swap_pairs = [
        ('nodeid_1', 'nodeid_2'),
        ('auth_chain1', 'auth_chain2'),
        ('resn1', 'resn2'),
        ('uniprot_resi1', 'uniprot_resi2'),
        ('res3n1', 'res3n2'),
        ('pdb_raw_resi1', 'pdb_raw_resi2'), 
        ('uniprot1', 'uniprot2'),
        ('pdb_auth_resi1', 'pdb_auth_resi2'),
        ('remove_homo_uniprot1', 'remove_homo_uniprot2')
    ]
    
    energy_cols = [
        'coulombs_energy', 
        'lj_energy', 
        'total_energy', 
        'cleaned_lj_energy', 
        'cleaned_total_energy'
    ]
    
    mask = df['nodeid_1'] > df['nodeid_2']
    
    for col1, col2 in swap_pairs:
        if col1 in df.columns and col2 in df.columns:
            df.loc[mask, [col1, col2]] = df.loc[mask, [col2, col1]].values

    group_cols = ['nodeid_1', 'nodeid_2']
    
    agg_strategy = {}
    
    for col in df.columns:
        if col in group_cols:
            continue
        if col in energy_cols:
            agg_strategy[col] = 'mean'
        else:
            agg_strategy[col] = 'first'

    merged_df = df.groupby(group_cols, as_index=False).agg(agg_strategy)

    assert merged_df.duplicated().sum() == 0, "Duplicate edges found after merging energy nodes"
    
    removed_cols = ['auth_chain1', 'auth_chain2', 'pdb_code', 'source', 'uniprot1', 'uniprot2']
    cols_to_drop = [c for c in removed_cols if c in merged_df.columns]
    merged_df = merged_df.drop(columns=cols_to_drop)
    
    print(f"Merge energy nodes: [{len(df)} -> {len(merged_df)}] -> Removed {len(df) - len(merged_df)} edges")
    
    return merged_df

def remove_duplicate_edges(df, subset=['nodeid_1', 'nodeid_2']):
    initial_count = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    print(f"Remove duplicate edges: [{initial_count} -> {len(df)}] -> Removed {initial_count - len(df)} edges")
    return df
    
def remove_NaN_in_energy(df, energy_col='cleaned_total_energy'):
    initial_count = len(df)
    df = df.dropna(subset=[energy_col]).reset_index(drop=True)
    print(f"Remove NaN in energy: [{initial_count} -> {len(df)}] -> Removed {initial_count - len(df)} edges")
    return df

def GenerateGraph_from_edge(edge_df,src='nodeid_1', tar='nodeid_2', weight_col='cleaned_total_energy'):
    G = nx.from_pandas_edgelist(edge_df, 
                                source=src, 
                                target=tar,
                                edge_attr=[weight_col]
                                )
    return G