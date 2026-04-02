#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=add_pssm
#SBATCH --output=logs/add_pssm_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:00:00

# load environment variables
source ~/.bashrc

# move to base PATH
cd YOUR_PROJECT_DIR

# Load module
module load StdEnv/2023
module load openmpi/5.0.8

source YOUR_ENV_DIR

# Add tools to PATH
export PATH=$PATH:YOUR_TOOL_DIR/ncbi-blast-2.17.0+/bin
export PATH=$PATH:YOUR_TOOL_DIR/hhsuite-3.3.0-SSE2/bin

# Run the script
srun python data/generation/evol/evolExtractor.py \
     --fasta_dir FASTA_DIR \
     --nr_db_path NR_DB_DIR \
     --workers 4 \
     --pssm_dir PSSM_DIR \
     --jobs pssm 