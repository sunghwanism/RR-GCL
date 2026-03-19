from collections import Counter
import ast
import pandas as pd

CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def translate_dna(codon):
    return CODON_TABLE.get(codon, 'X')

def get_nonsynonymous_mutability(five_mer, mutation_freq_dict):
    
    if len(five_mer) != 5:
        return 0.0

    b1, b2, b3, b4, b5 = list(five_mer)
    original_codon = b2 + b3 + b4
    original_aa = translate_dna(original_codon)
    
    mutability_score = 0.0

    context1 = b1 + b2 + b3
    if context1 in mutation_freq_dict:
        for mutated_seq, freq in mutation_freq_dict[context1].items():
            mutated_b2 = mutated_seq[1] # 가운데 글자가 변이된 염기
            new_codon = mutated_b2 + b3 + b4
            if translate_dna(new_codon) != original_aa:
                mutability_score += freq

    context2 = b2 + b3 + b4
    if context2 in mutation_freq_dict:
        for mutated_seq, freq in mutation_freq_dict[context2].items():
            mutated_b3 = mutated_seq[1]
            new_codon = b2 + mutated_b3 + b4
            if translate_dna(new_codon) != original_aa:
                mutability_score += freq

    context3 = b3 + b4 + b5
    if context3 in mutation_freq_dict:
        for mutated_seq, freq in mutation_freq_dict[context3].items():
            mutated_b4 = mutated_seq[1]
            new_codon = b2 + b3 + mutated_b4
            if translate_dna(new_codon) != original_aa:
                mutability_score += freq
                
    return mutability_score


def calculate_mutability_for_row(row, mutation_freq):
    if pd.isna(row.get('unique_cds_contexts')):
        return None

    try:
        val_unique = row['unique_cds_contexts']
        if isinstance(val_unique, str):
            unique_cds_context = list(ast.literal_eval(val_unique))
        else:
            unique_cds_context = list(val_unique)
            
        val_full = row['cds_contexts']
        if isinstance(val_full, str):
            cds_contexts = list(ast.literal_eval(val_full))
        else:
            cds_contexts = list(val_full)
            
    except (ValueError, SyntaxError):
        return None

    cds_counts = Counter(cds_contexts)
    total_count = len(cds_contexts)
    
    if total_count == 0:
        return 0.0

    total_mutability_sum = 0.0
    
    for cds in unique_cds_context:
        context_mutability = get_nonsynonymous_mutability(cds, mutation_freq)
        weight = cds_counts[cds] / total_count
        total_mutability_sum += (context_mutability * weight)

    return total_mutability_sum