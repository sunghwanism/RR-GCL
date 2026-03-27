#!/bin/bash
#SBATCH --account=ADD_ACCOUNT
#SBATCH --job-name=AM_Processing
#SBATCH --output=logs/AM_Processing_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=ADD_NUM_CORRES
#SBATCH --mem=32G
#SBATCH --time=01:00:00


# Load bashrc
source ~/.bashrc

# move to base PATH
cd YOUR_PROJECT_DIR

# Load module
module load StdEnv/2023
module load openmpi/5.0.8

source YOUR_ENV_DIR/YOUR_ENV_NAME/bin/activate

srun python -m data.generation.alphamissense.amProcessing \
    --amPATH YOUR_TSVGZ_PATH_OF_AM \
    --graphPATH data/proc_data/YOUR_GRAPH_PKL \
    --savePATH data/proc_data/ \
    --num_workers YOUR_NUM_WORKERS # Same with cpu-per-task