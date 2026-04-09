import pandas as pd
import numpy as np
import os
import requests
import warnings
import gzip
import shutil
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm


# --- Helper Functions ---

def get_auth_to_label_mapping(pdb_code): # change from get_tab_chain_mapping
    """
    Queries the RCSB PDB API to create a mapping:
    Auth Chain (auth_asym_ids) -> Label Chain (asym_ids)
    """
    pdb_code = pdb_code.upper()
    mapping = {}
    
    # 1. Fetch the list of all Entity IDs for the given PDB code
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code}"
    try:
        entry_res = requests.get(entry_url)
        entry_res.raise_for_status()
        entry_data = entry_res.json()
        
        # Extract polymer_entity_ids (e.g., ["1", "2", "3"])
        entity_ids = entry_data.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])
        
        # 2. Iterate through each Entity ID to get detailed mapping info
        for e_id in entity_ids:
            entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_code}/{e_id}"
            entity_res = requests.get(entity_url)
            
            if entity_res.status_code == 200:
                entity_data = entity_res.json()
                
                # Navigate to the container identifiers as found in your JSON sample
                container = entity_data.get('rcsb_polymer_entity_container_identifiers', {})
                
                auth_list = container.get('auth_asym_ids', []) # e.g., ["F", "G", "H", "J"]
                label_list = container.get('asym_ids', [])      # e.g., ["A", "B", "C", "D"]
                
                # Pair the lists 1:1 and add to the dictionary
                for auth, label in zip(auth_list, label_list):
                    mapping[str(auth)] = str(label)
                    
        return mapping

    except Exception as e:
        print(f"Error generating mapping for {pdb_code}: {e}")
        return {}


def get_pdb_redo_cif(pdb_code, args):
    """
    Downloads the DSSP-enriched mmCIF from PDB-REDO.
    """
    pdb_code = pdb_code.lower()
    dest_path = os.path.join(args.tmp_dir, f"{pdb_code}_dssp.cif")
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
        
    url = f"https://pdb-redo.eu/dssp/db/{pdb_code}/mmcif"
    try:
        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(r.content)
            return dest_path
        return None
    except Exception as e:
        print(f"Download Error {pdb_code}: {e}")
        return None

def parse_dssp_data_from_cif(cif_path):
    """
    Parses the _dssp_struct_summary category from the mmCIF file.
    Returns: {(label_chain, label_seq_id): { 'field_name': value, ... }}
    """
    try:
        mmcif_dict = MMCIF2Dict(cif_path)
    except Exception:
        return {}

    fields_to_extract = {
        'dssp_sec_struct': 'secondary_structure',
        'dssp_ss_bridge': 'ss_bridge',
        'dssp_helix_3_10': 'helix_3_10',
        'dssp_helix_alpha': 'helix_alpha',
        'dssp_helix_pi': 'helix_pi',
        'dssp_helix_pp': 'helix_pp',
        'dssp_bend': 'bend',
        'dssp_chirality': 'chirality',
        'dssp_sheet': 'sheet',
        'dssp_strand': 'strand',
        'dssp_ladder_1': 'ladder_1',
        'dssp_ladder_2': 'ladder_2',
        'dssp_accessibility': 'accessibility',
        'dssp_TCO': 'TCO',
        'dssp_kappa': 'kappa',
        'dssp_alpha': 'alpha',
        'dssp_phi': 'phi',
        'dssp_psi': 'psi'
    }

    try:
        chains = mmcif_dict.get('_dssp_struct_summary.label_asym_id', [])
        seq_ids = mmcif_dict.get('_dssp_struct_summary.label_seq_id', [])
        comp_ids = mmcif_dict.get('_dssp_struct_summary.label_comp_id', []) 
        
        if isinstance(chains, str): chains = [chains]
        if isinstance(seq_ids, str): seq_ids = [seq_ids]
        if isinstance(comp_ids, str): comp_ids = [comp_ids]
        
    except KeyError:
        return {} 

    data_columns = {}
    for out_name, suffix in fields_to_extract.items():
        full_key = f"_dssp_struct_summary.{suffix}"
        val_list = mmcif_dict.get(full_key, [])
        if isinstance(val_list, str): val_list = [val_list]
        
        if not val_list or len(val_list) != len(chains):
            data_columns[out_name] = [None] * len(chains)
        else:
            data_columns[out_name] = val_list

    dssp_map = {}
    float_cols = ['dssp_accessibility', 'dssp_TCO', 'dssp_kappa', 'dssp_alpha', 'dssp_phi', 'dssp_psi']

    for i in range(len(chains)):
        try:
            # Key: (Label Chain, Label Seq ID)
            # Chain "NA" will be preserved as string "NA" here by MMCIF2Dict
            key = (chains[i], int(seq_ids[i]))
            
            entry = {'aa_type': comp_ids[i]} 
            
            for out_name in fields_to_extract:
                raw_val = data_columns[out_name][i]
                
                if raw_val in ['.', '?', '', None]:
                    entry[out_name] = None
                else:
                    if out_name in float_cols:
                        try:
                            entry[out_name] = float(raw_val)
                        except ValueError:
                            entry[out_name] = None
                    else:
                        entry[out_name] = raw_val
            
            dssp_map[key] = entry
            
        except (ValueError, IndexError):
            continue
            
    return dssp_map

def get_center_of_mass(chain):
    coords = [res['CA'].get_coord() for res in chain if 'CA' in res]
    if not coords: return np.array([0,0,0])
    return np.mean(coords, axis=0)

def compute_hse(residue, neighbors, radius=10.0):
    if 'CA' not in residue: return None, None
    ca_vec = residue['CA'].get_vector()
    
    if residue.get_resname() == 'GLY':
        if 'N' not in residue: return None, None
        ref_vec = residue['N'].get_vector() - ca_vec
    else:
        if 'CB' not in residue: return None, None
        ref_vec = residue['CB'].get_vector() - ca_vec
    
    ref_vec.normalize()
    up, down = 0, 0
    
    for n in neighbors:
        if n == residue or 'CA' not in n: continue
        n_vec = n['CA'].get_vector() - ca_vec
        if n_vec.norm() > radius: continue
        if ref_vec * n_vec > 0: up += 1
        else: down += 1
    return up, down

def process_pdb(pdb_code, target_residues, cif_file,
                MAX_SASA, SEC_STRUCT_MAP):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_code, cif_file)
        model = structure[0]
        chain_coms = {c.id: get_center_of_mass(c) for c in model}
        atoms = [a for a in model.get_atoms() if a.get_name() == 'CA']
        ns = NeighborSearch(atoms)
    except Exception:
        model = None
        ns = None

    dssp_lookup = parse_dssp_data_from_cif(cif_file)
    auth_to_label = get_auth_to_label_mapping(pdb_code)
    
    results = []
    
    for auth_chain, auth_pos, raw_pos in target_residues:
        
        # --- A. Geometric Features ---
        depth, hse_up, hse_down = None, None, None
        if model:
            try:
                res_id_auth = (' ', int(auth_pos), ' ')
                if auth_chain in model and res_id_auth in model[auth_chain]:
                    residue_obj = model[auth_chain][res_id_auth]

                    if 'CA' in residue_obj and auth_chain in chain_coms:
                        depth = np.linalg.norm(residue_obj['CA'].get_coord() - chain_coms[auth_chain])

                    if 'CA' in residue_obj:
                        nbs = ns.search(residue_obj['CA'].get_coord(), 10.0, level='R')
                        hse_up, hse_down = compute_hse(residue_obj, nbs)
                        
            except Exception:
                pass

        # --- B. DSSP Features ---
        row_data = {
            'pdb_code': pdb_code,
            'chain_id': auth_chain,
            'label_chain_id': None,
            'auth_residue_number': auth_pos,
            'pdb_raw_resi': raw_pos,
            'rel_sasa': None,
            'ss_helix': 0, 'ss_sheet': 0, 'ss_loop': 0,
            'dssp_quality': "Missing_Chain_Map",
            'depth': depth, 'hse_up': hse_up, 'hse_down': hse_down
        }
        
        extra_dssp_cols = [
            'dssp_sec_struct', 'dssp_ss_bridge', 'dssp_helix_3_10', 'dssp_helix_alpha', 
            'dssp_helix_pi', 'dssp_helix_pp', 'dssp_bend', 'dssp_chirality', 'dssp_sheet', 
            'dssp_strand', 'dssp_ladder_1', 'dssp_ladder_2', 'dssp_accessibility', 
            'dssp_TCO', 'dssp_kappa', 'dssp_alpha', 'dssp_phi', 'dssp_psi'
        ]
        for col in extra_dssp_cols:
            row_data[col] = None

        label_chain = auth_to_label.get(auth_chain)
        row_data['label_chain_id'] = label_chain
        
        if label_chain:
            key = (label_chain, int(raw_pos))
            dssp_data = dssp_lookup.get(key)
            
            if dssp_data:
                for col in extra_dssp_cols:
                    row_data[col] = dssp_data.get(col)
                
                raw_sasa = dssp_data.get('dssp_accessibility')
                raw_ss = dssp_data.get('dssp_sec_struct')
                aa_type = dssp_data.get('aa_type', 'GLY')

                ss_cat = SEC_STRUCT_MAP.get(raw_ss, 'Loop')
                row_data['ss_helix'] = 1 if ss_cat == 'Helix' else 0
                row_data['ss_sheet'] = 1 if ss_cat == 'Sheet' else 0
                row_data['ss_loop']  = 1 if ss_cat == 'Loop'  else 0
                
                if raw_sasa is not None:
                    max_s = MAX_SASA.get(aa_type, 200.0)
                    val = raw_sasa / max_s
                    if val > 1.2:
                        row_data['dssp_quality'] = "Unreliable_High"
                        row_data['rel_sasa'] = 1.0
                    elif val < 0:
                        row_data['dssp_quality'] = "Unreliable_Neg"
                        row_data['rel_sasa'] = 0.0
                    else:
                        row_data['dssp_quality'] = "Reliable"
                        row_data['rel_sasa'] = min(1.0, val)
                else:
                    row_data['dssp_quality'] = "Missing_SASA"
            else:
                row_data['dssp_quality'] = "Missing_Residue_In_DSSP"
        
        results.append(row_data)
        
    return results