#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=gs_cluster_residue
#SBATCH --output=logs/gs_cluster_residue_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=05:00:00

# load environment variables
source ~/.bashrc

# move to base PATH
cd $PROJECT_ROOT

# Load module
module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

source $ENV_DIR/rrgcl/bin/activate

# Run the script
# srun python models/RRGCL/src/mutprofile/cluster.py \
#     --target neighbor \
#     --feat_path data/proc_data/aug_ngb_orthogonal_feature_data_for_clustering.csv \
#     --n_jobs 64 \
#     --grid_sc

srun python models/RRGCL/src/mutprofile/cluster.py \
    --target residue \
    --feat_path data/proc_data/aug_res_feature_data_for_clustering.csv \
    --n_jobs 64 \
    --grid_sc