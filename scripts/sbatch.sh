#!/bin/bash
#SBATCH --account=ADD_ACCOUNT_ID
#SBATCH --job-name=SanityCheck
#SBATCH --output=logs/SanityCheck_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=11:59:00


# GPU Selection
# h100:1
# nvidia_h100_80gb_hbm3_2g.20gb:1
# nvidia_h100_80gb_hbm3_3g.40gb:1
# nvidia_h100_80gb_hbm3_1g.10gb:1

# Load bashrc
source ~/.bashrc

# move to base PATH
cd $SCRATCH/$USER/RR-GCL

# Load module
module load StdEnv/2023 nvhpc/25.1 openmpi/5.0.3
module load cuda/12.6

srun nproc
srun nvidia-smi

source $ENV_DIR/RR-GCL/bin/activate

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# NEED to set for wandb
export WANDB_RUN_NAME="SanityCheck"
export WANDB_API_KEY=$WANDB_API # Add Your WandB API Key
export ENTITY_NAME=$WANDB_ENTITY # Add Your WandB Entity Name

# [Important]
# If you want to load_pretrained model, you need to set WANDB_RUN_ID
# export WANDB_RUN_ID=""

srun python models/train.py \
     --config_path config/RRGCL.yaml \
     --wandb_key $WANDB_API_KEY \
     --entity_name $ENTITY_NAME \
     --project_name RR-GCL \
     --wandb_run_name $WANDB_RUN_NAME \
     --batch_size 256 \
     --num_workers 2 \
     --use_aug
     # --load_pretrained \
     # --wandb_run_id $WANDB_RUN_ID \