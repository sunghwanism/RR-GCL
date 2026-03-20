import os
import sys
import argparse

import numpy as np
import pandas as pd

from utils.graphfunction import load_graph, get_sample

from data.generation.evol.helper import *


def LoadParser():
    parser = argparse.ArgumentParser(description='Process HMM & PSSM files.')
    parser.add_argument('--evolPATH', type=str, required=True, help='Path to HMM&PSSM parent files.')
    parser.add_argument('--savePATH', type=str, required=True, help='Path to save processed files.')
    return parser.parse_args()


def main(args):

    # HMM Processing (hhm -> json File) [emission 20 & transition 7 & neff 1]
    # Amino Acid Order: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    # {node_id: [ProbA, ProbC, ProbD, ..., ProbY, M->M, M->I, M->D, I->M, I->I, D->M, D->D, Neff],
    #  node_id: [ProbA, ProbC, ProbD, ..., ProbY, M->M, M->I, M->D, I->M, I->I, D->M, D->D, Neff],
    #  ...}
    hmm_dir = os.path.join(args.evolPATH, 'hmm')
    hmm_json = os.path.join(args.savePATH, 'hmm_features.json')
    merge_hmm_to_json(hmm_dir, hmm_json)
    
    # PSSM Processing (pssm -> json File) [log-odd score & entropy]
    # Amino Acid Order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
    # {node_id: [LogOddA, LogOddR, ..., LogOddV, Entropy],
    #  node_id: [LogOddA, LogOddR, ..., LogOddV, Entropy],
    #  ...}
    pssm_dir = os.path.join(args.evolPATH, 'pssm')
    pssm_json = os.path.join(args.savePATH, 'pssm_features.json')
    merge_pssm_to_json(pssm_dir, pssm_json)

if __name__ == '__main__':
    args = LoadParser()
    main(args)