import pandas as pd
import os
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Matching SS to node_id')
    parser.add_argument('--ss_df', type=str, required=True, help='SS DataFrame (csv)') # ss_info 
    parser.add_argument('--edge_df', type=str, help='Edge DataFrame (csv)')
    parser.add_argument('--save_dir', type=str, help='save path and file name')

    return parser.parse_args()

def read_csv_with_progress(file_path, desc="Reading file"):
    """
    Reads a CSV file in chunks and displays a progress bar.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    # Get total lines for the progress bar
    # (This takes a moment but allows the bar to show a % completion)
    with open(file_path, "rb") as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract header

    chunksize = 50000
    chunks = []
    
    # Read with tqdm wrapper
    with tqdm(total=total_lines, desc=desc, unit="rows") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            chunks.append(chunk)
            pbar.update(len(chunk))
            
    return pd.concat(chunks, axis=0)


def main(args):

    # 1. Load the Computed Features
    df_features = read_csv_with_progress(args.ss_df, desc="Loading Features")
    if df_features is None: return
    
    # Deduplicate features just in case
    df_features = df_features.drop_duplicates(subset=['pdb_code', 'chain_id', 'auth_residue_number'])

    # 2. Load the Contact Table
    df_contacts = read_csv_with_progress(args.edge_df, desc="Loading Contacts")
    if df_contacts is None: return
    
    # 3. Extract Mapping: (PDB, Chain, Res) -> NodeID
    # We use a progress bar here manually for the two-step concatenation
    with tqdm(total=3, desc="Mapping Nodes", unit="step") as pbar:
        
        # Extract Left Side
        map_1 = df_contacts[['pdb_code', 'auth_chain1', 'pdb_auth_resi1', 'nodeid_1']].rename(
            columns={'auth_chain1': 'chain_id', 'pdb_auth_resi1': 'auth_residue_number', 'nodeid_1': 'node_id'}
        )
        pbar.update(1)
        
        # Extract Right Side
        map_2 = df_contacts[['pdb_code', 'auth_chain2', 'pdb_auth_resi2', 'nodeid_2']].rename(
            columns={'auth_chain2': 'chain_id', 'pdb_auth_resi2': 'auth_residue_number', 'nodeid_2': 'node_id'}
        )
        pbar.update(1)
        
        # Combine and Deduplicate
        node_map = pd.concat([map_1, map_2]).drop_duplicates(subset=['pdb_code', 'chain_id', 'auth_residue_number', 'node_id'])
        pbar.update(1)

    print(f"   > Found {len(node_map)} unique NodeIDs.")

    # 4. Data Type Conversion (Crucial for successful merging)
    print("\nStep 3: Merging...")
    
    # Ensure join keys are strings to avoid type mismatches (e.g., '10' vs 10)
    for df in [node_map, df_features]:
        df['pdb_code'] = df['pdb_code'].astype(str).str.lower()
        df['chain_id'] = df['chain_id'].astype(str)
        df['auth_residue_number'] = df['auth_residue_number'].astype(str)

    # Merge
    # We use 'left' join to keep all nodes from the contact table, 
    # even if features are missing (they will appear as NaN)
    final_df = pd.merge(
        node_map,
        df_features,
        on=['pdb_code', 'chain_id', 'auth_residue_number'],
        how='left'
    )

    # 5. Save Output
    print(f"\nStep 4: Saving {len(final_df)} rows to disk...")
    final_df.to_csv(args.save_dir, index=False)
    
    # Summary
    filled = final_df['rel_sasa'].notna().sum()
    print("-" * 40)
    print("COMPLETED SUCCESSFULLY")
    print(f"Total Nodes mapped:     {len(final_df)}")
    print(f"Nodes with features:    {filled} ({filled/len(final_df):.1%})")
    print(f"Nodes missing features: {len(final_df) - filled}")
    print(f"Output file:            {args.save_dir}")
    print("-" * 40)


if __name__ == "__main__":
    args = get_args()
    main(args)