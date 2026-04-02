import os
from typing import List, Union

import numpy as np
import pandas as pd

from utils.graphfunction import get_res_from_nodes, get_node_id_rm_copy
from data.reference import residue3to1


#################################
# AAindex Processing Helper
#################################

def get_aainfo(node_id: str, aaData: Union[pd.DataFrame, List[pd.DataFrame]]) -> List[float]:
    """
    Extract amino acid information from node_id and return corresponding AAindex data.
    
    Args:
        node_id: Node ID in 'UniProtID_Position_ResType' format.
        aaData: Amino acid index data (either a DataFrame or a list of DataFrames).
        
    Returns:
        A list of extracted property values.
    """
    # 1. Extract residue type from node_id (e.g., 'glu' -> 'GLU')
    res_type_3 = get_res_from_nodes(node_id).upper()
    
    # 2. Convert 3-letter code to 1-letter code (e.g., 'GLU' -> 'E')
    res_type_1 = residue3to1.get(res_type_3, 'X')
    
    # 3. Retrieve data
    if isinstance(aaData, list):
        # If it's a list, concatenate rows from all DataFrames for the given residue
        all_features = []
        for df in aaData:
            if res_type_1 in df.index:
                all_features.extend(df.loc[res_type_1].tolist())
            else:
                # Padding with zeros if residue not found
                all_features.extend([0.0] * len(df.columns))
        return all_features
    else:
        # If it's a single DataFrame
        if res_type_1 in aaData.index:
            return aaData.loc[res_type_1].tolist()
        else:
            return [0.0] * len(aaData.columns)