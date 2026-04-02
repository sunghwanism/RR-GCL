#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=add_hmm
#SBATCH --output=logs/add_hmm_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00

# load environment variables
source ~/.bashrc

# move to base PATH
cd YOUR_PROJECT_NAME

# Load module
module load StdEnv/2023
module load openmpi/5.0.8

source YOUR_ENV_DIR/bin/activate

# Add tools to PATH
export PATH=$PATH:YOUR_TOOL_DIR/ncbi-blast-2.17.0+/bin
export PATH=$PATH:YOUR_TOOL_DIR/hhsuite-3.3.0-SSE2/bin

# Run the script
srun python data/generation/evol/postProcessing.py \
     --fasta_dir FASTA_DIR \
     --uniref_db_path UniRef30_DIR \
     --workers 4 \
     --hmm_dir HMM_DIR \
     --hhm_output_dir HHM_OUTPUT_DIR \
     --jobs hmm