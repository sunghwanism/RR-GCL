# Data Generation & Processing

## Graph Generation

### Require Files
#### DataFolder (BASEPATH)
1. [CSV] Residue-Residue contact table (Edge Table)

#### Reference Folder (BASEPATH/reference)
1. [CSV-1column] Specific PDB or UniProt ID Table you want to include or exclude (optional)
2. [CSV] CDS Context Table

#### Save Folder (BASEPATH/processing & BASEPATH/final)
- Intermediate processing results are saved in ```processing``` Folder
- Final results (cleaned edge table and graph) are saved in ```final``` Folder

### Graph Generation Process
1. Enter configuration in ```processing.yaml```
2. Load Edge Table
3. Apply filtering conditions (PDB, UniProt, Node)
4. Remove Strange Position (zero, negative, nan)
5. Align CDS Context Table to get Background Mutability (Optional)
6. Generate Inter-Chain w/ homodimer & w/o homodimer Graph
7. Remove Duplicated Node Pair & Merge Energy from Edge Table
8. Graph Generation (pkl files)

## Evolutionary Information
### Require Files
#### HMM Files
- Results of ```PSSM``` generated using PSI-BLAST (BLOSUM62, default Parameters)
- Results of ```HMM``` generated using HHblits (UniRef30_2024_03, default Parameters)
#### Preprocessing
```sh
python -m data.generation.evol.evolProcessing --evolPATH <Evol_PATH> --savePATH <SAVE_PATH>
```
#### Feature Dimension
- HMM
  - Emission: 20
  - Transition: 7
  - Neff: 1
- PSSM
  - Log-odd score: 20
  - Entropy: 1

