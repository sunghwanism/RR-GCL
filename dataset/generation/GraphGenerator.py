import os

import argparse

import json
from datetime import datetime

from helper import *

from utils.functions import load_yaml
from utils.graphfunction import get_unique_node

DATE = datetime.now().strftime("%m%d%y")

def main(args):
    EDGEPATH = os.path.join(args.BASEPATH, args.EDGEFILE)
    INTERMEDIATEPATH = os.path.join(args.BASEPATH, 'processing')
    FINALPATH = os.path.join(args.BASEPATH, 'final')
    REFPATH = os.path.join(args.BASEPATH, args.REFPATH)

    # Check & Make Dirs
    check_path(EDGEPATH, INTERMEDIATEPATH, FINALPATH, REFPATH)

    # Load condition files (PDB, UniProt)
    print("[Before Start] Loading condition files (PDB, UniProt, node)")
    (incl_pdb, incl_uniprot, incl_node,
     except_pdb, except_uniprot, except_node) = load_pdb_uniprot_node(args)

    # Load edge file
    print("[Start] Loading edge file")
    EDGES = pd.read_csv(EDGEPATH)
    print(f"[Start] Total Nodes: {len(get_unique_node(EDGES))}, Total Edges: {len(EDGES)}")

    # Step 1: Apply filtering conditions || To-do: Add incl_pdb, incl_node, except_node
    print("[Step 1] Applying filtering conditions (PDB, UniProt, Node)")
    EDGES = generate_nodeid_and_only_uniprot(EDGES)
    EDGES = remove_ubq_related_connection(EDGES, except_uniprot)
    EDGES = filter_only_nucleosome_related_connection(EDGES, except_pdb, incl_uniprot)

    usable_nodes_set = set(incl_node)
    cond1 = EDGES['nodeid_1'].str.replace(r'-\d+', '', regex=True).isin(usable_nodes_set)
    cond2 = EDGES['nodeid_2'].str.replace(r'-\d+', '', regex=True).isin(usable_nodes_set)
    EDGES = EDGES[cond1 & cond2]
    del usable_nodes, cond1, cond2, usable_nodes_set

    print(f"[Step 1] Total Nodes: {len(get_unique_node(EDGES))}, Total Edges: {len(EDGES)}")
    EDGES.to_csv(os.path.join(INTERMEDIATEPATH, 
                              f'step1_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv'), 
                              index=False)

    # Step 2: Remove Strange Position
    EDGES = remove_negative_and_zero_position_node(EDGES)
    print(f"[Step 2] Total Nodes: {len(get_unique_node(EDGES))}, Total Edges: {len(EDGES)}")
    EDGES.to_csv(os.path.join(INTERMEDIATEPATH, 
                              f'step2_rmStrangeEdge_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv'), 
                              index=False) 
    
    # Step 3: Apply CDS Filtering (if CDS_FILTER=True)
    if args.CDS_FILTER:
        CDSTable = pd.read_csv(os.path.join(REFPATH, args.CDSFILE))
        CDSTable = CDSTable.dropna(subset='unique_cds_contexts', axis=0)

        CDSTable = ready_for_CDS_filter(args, CDSTable, cancer_type=args.CANCERTYPE)

        print("Save Mutability Rate with CDS: ", os.path.join(INTERMEDIATEPATH, f"background_mutability_v{DATE}.csv"))
        CDSTable.to_csv(os.path.join(args.BASEPATH, f"background_mutability_v{DATE}.csv"), index=False)

        applicable_node = CDSTable.node_id.tolist()
        EDGES = EDGES[EDGES['nodeid_1'].isin(applicable_node) & 
                    EDGES['nodeid_2'].isin(applicable_node)]

        del applicable_node, CDSTable

        print(f"[Step 3] Total Nodes: {len(get_unique_node(EDGES))}, Total Edges: {len(EDGES)}")
        EDGES.to_csv(os.path.join(INTERMEDIATEPATH, 
                                f'step3_only_mutability_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv'), 
                                index=False)

    # Step 4: Inter-Chain w/ homodimer & w/o homodimer
    inter_chain_w_homo = EDGES[EDGES['chain_flag']=='inter-chain']
    print(f"[Step 4 - Incl Homodimer] Total Nodes: {len(get_unique_node(inter_chain_w_homo))}, Total Edges: {len(inter_chain_w_homo)}")
    if args.CDS_FILTER:
        FileName = f'step4_inter-chain_only_inclHomodimer_mutability_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv'
    else:
        FileName = f'step4_inter-chain_inclHomodimer_exceptCDSFilter_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv'
    inter_chain_w_homo.to_csv(os.path.join(INTERMEDIATEPATH, FileName), index=False)

    inter_chain_wo_homo = inter_chain_w_homo[inter_chain_w_homo['remove_homo_uniprot1'] != inter_chain_w_homo['remove_homo_uniprot2']]
    print(f"[Step 4 - Excl Homodimer] Total Nodes: {len(get_unique_node(inter_chain_wo_homo))}, Total Edges: {len(inter_chain_wo_homo)}")
    if args.CDS_FILTER:
        FileName = f'step4_inter-chain_only_exclHomodimer_mutability_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv'
    else:
        FileName = f'step4_inter-chain_exclHomodimer_exceptCDSFilter_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv'
    inter_chain_wo_homo.to_csv(os.path.join(INTERMEDIATEPATH, FileName), index=False)

    # Step 5: Remove Duplicated Node Pair & Merge Energy
    ## 1) Inter+Intra Graph
    EDGES = remove_NaN_in_energy(EDGES)
    EDGES = merge_energy_nodes(EDGES)

    if args.CDS_FILTER:
        FileName = f"step5_all-chain_only_mutability_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv"
    else:
        FileName = f"step5_all-chain_exceptCDSFilter_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv"
    EDGES.to_csv(os.path.join(FINALPATH, FileName), index=False)
    
    ## 2) Inter-chain Graph
    ### 2-1) Incl Homodimer
    inter_chain_w_homo = remove_NaN_in_energy(inter_chain_w_homo)
    inter_chain_w_homo = merge_energy_nodes(inter_chain_w_homo)

    if args.CDS_FILTER:
        FileName = f"step5_inter-chain_inclHomodimer_mutability_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv"
    else:
        FileName = f"step5_inter-chain_inclHomodimer_exceptCDSFilter_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv"
    inter_chain_w_homo.to_csv(os.path.join(FINALPATH, FileName), index=False)

    ### 2-2) Excl Homodimer
    inter_chain_wo_homo = remove_NaN_in_energy(inter_chain_wo_homo)
    inter_chain_wo_homo = merge_energy_nodes(inter_chain_wo_homo)

    if args.CDS_FILTER:
        FileName = f"step5_inter-chain_exclHomodimer_mutability_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv"
    else:
        FileName = f"step5_inter-chain_exclHomodimer_exceptCDSFilter_human_aa_edges_exclubq_Nucleosome_related_data_v{DATE}.csv"
    inter_chain_wo_homo.to_csv(os.path.join(FINALPATH, FileName), index=False)

    # Step 6: Graph Generation

    for i, edge_df in enumerate([EDGES, inter_chain_w_homo, inter_chain_wo_homo]):
        G = GenerateGraph_from_edge(edge_df, 
                                    src='nodeid_1',
                                    tar='nodeid_2',
                                    weight_col=args.WEIGHT_COL
                                    )
        if i == 0:
            print("[Final] Graph Generation (All-chain)")
            print(f"Total Nodes: {len(G.nodes)}, Total Edges: {len(G.edges)}")
            if args.CDS_FILTER:
                GraphSAVE = os.path.join(args.SAVEPATH, f'weighted_all-chain_incl_homodimer_v{DATE}.pkl')
            else:
                GraphSAVE = os.path.join(args.SAVEPATH, f'skipCDSFilter_weighted_all-chain_v{DATE}.pkl')

        elif i == 1:
            print("[Final] Graph Generation (Inter-chain w/ homodimer)")
            print(f"Total Nodes: {len(G.nodes)}, Total Edges: {len(G.edges)}")
            if args.CDS_FILTER:
                GraphSAVE = os.path.join(args.SAVEPATH, f'weighted_inter-chain_incl_homodimer_v{DATE}.pkl')
            else:
                GraphSAVE = os.path.join(args.SAVEPATH, f'skipCDSFilter_weighted_inter-chain_incl_homodimer_v{DATE}.pkl')

        elif i == 2:
            print("[Final] Graph Generation (Inter-chain w/o homodimer)")   
            print(f"Total Nodes: {len(G.nodes)}, Total Edges: {len(G.edges)}")
            if args.CDS_FILTER:
                GraphSAVE = os.path.join(args.SAVEPATH, f'weighted_inter-chain_excl_homodimer_v{DATE}.pkl')
            else:
                GraphSAVE = os.path.join(args.SAVEPATH, f'skipCDSFilter_weighted_inter-chain_incl_homodimer_v{DATE}.pkl')

        with open(GraphSAVE, 'wb') as f:
            pickle.dump(G, f)
                            

if __name__ == "__main__":
    args = load_yaml("processing.yaml")
    main(args)
    