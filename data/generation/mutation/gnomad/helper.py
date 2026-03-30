
import re
import pandas as pd


from data.reference import residue1to3, residue3to1

def process_gnomad_csv(df_raw, uniprot_id, enst_id, gene_symbol=None):
    full_enst = df_raw['Transcript'].iloc[0]
    enst_id = full_enst.split('.')[0] if '.' in full_enst else full_enst

    def parse_hgvsp(hgvsp):
        if not isinstance(hgvsp, str): return None, None, None
        match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', hgvsp)
        if match:
            ref_aa = residue3to1.get(match.group(1).upper())
            pos = int(match.group(2))
            alt_aa = residue3to1.get(match.group(3).upper())
            return ref_aa, pos, alt_aa
        return None, None, None

    df = df_raw[df_raw['VEP Annotation'] == 'missense_variant'].copy()
    df['Protein Consequence'] = df['Protein Consequence']
    parsed = df['Protein Consequence'].apply(parse_hgvsp)
    df['ref_1'], df['position'], df['alt_1'] = zip(*parsed)
    df = df.dropna(subset=['position'])
    df['position'] = df['position'].astype(int)

    df['uniprot_id'] = uniprot_id
    df['enst_id'] = enst_id
    df['gene_symbol'] = gene_symbol
    df['residuetype'] = df['ref_1'].map(lambda x: residue1to3[x]).str.lower()
    df['alt_aa_3'] = df['alt_1'].map(lambda x: residue1to3[x]).str.lower()

    counts = df.groupby(['uniprot_id', 'enst_id', 'residuetype', 'position']).agg(
        unique_mutation_type=('alt_aa_3', 'nunique'),
        total_mutations_count=('Allele Count', 'sum'),
        total_number=('Allele Number', 'max')
    ).reset_index()

    counts['frequency'] = counts['total_mutations_count'] / counts['total_number']

    variant_dict = df.groupby(['uniprot_id', 'enst_id', 'residuetype', 'position']).apply(
        lambda g: g.groupby('alt_aa_3')['Allele Count'].sum().to_dict()
    ).reset_index()
    variant_dict.rename({0:'variant'}, axis=1, inplace=True)
    
    result = counts.merge(variant_dict, on=['uniprot_id', 'enst_id', 'residuetype', 'position'])
    result['node_id'] = result.apply(
        lambda row: f"{str(row['uniprot_id']).lower()}_{int(row['position'])}_{row['residuetype']}", 
        axis=1
    )
    result['gene_symbol'] = gene_symbol
    cols = ['node_id', 'uniprot_id', 'enst_id', 'residuetype', 'position', 'gene_symbol',
            'unique_mutation_type', 'total_mutations_count', 'total_number', 'frequency', 'variant']
    
    return result[cols].sort_values('position').reset_index(drop=True)