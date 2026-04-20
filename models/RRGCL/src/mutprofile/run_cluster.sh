#!/bin/bash
#SBATCH --account=SLURM_ACCOUNT
#SBATCH --job-name=cluster_residue
#SBATCH --output=logs/cluster_residue_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# load environment variables
source ~/.bashrc

# move to base PATH
cd $PROJECT_DIR

# Load module
module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

source $ENV_DIR

# Run the script
srun python models/RRGCL/src/mutprofile/cluster.py --target residue --n_jobs 30