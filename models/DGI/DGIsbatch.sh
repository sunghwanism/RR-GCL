#!/bin/bash
#SBATCH --account=ACCOUNT_NAME
#SBATCH --job-name=DGI_train_1_all
#SBATCH --output=logs/DGI_train_1_all_j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=01:00:00


#############################
# GPU Selection
#############################
# h100:1
# nvidia_h100_80gb_hbm3_2g.20gb:1
# nvidia_h100_80gb_hbm3_3g.40gb:1
# nvidia_h100_80gb_hbm3_1g.10gb:1

# Load bashrc
source ~/.bashrc

# move to base PATH
cd $SCRATCH/$USER/RR-GCL

# Load module
module load StdEnv/2023
module load cuda/12.6

srun nproc
srun nvidia-smi

source $ENV_DIR/rrgcl/bin/activate

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# NEED to set for wandb
# export WANDB_RUN_NAME="SanityCheck"
export WANDB_API_KEY=$WANDB_API # Add Your WandB API Key
export ENTITY_NAME=$WANDB_ENTITY # Add Your WandB Entity Name

# [Important]
# If you want to load_pretrained model, you need to set WANDB_RUN_ID
# export WANDB_RUN_ID=""

FEATURES=(
    # Sequence Index
    copy_idx 
    # AAindex1
    aa1_KYTJ820101 aa1_KLEP840101 aa1_BHAR880101 aa1_JANJ780101 
    aa1_CHOP780201 aa1_GRAR740102 aa1_GRAR740103 
    # PSSM
    pssm_A pssm_C pssm_D pssm_E pssm_F pssm_G pssm_H pssm_I pssm_K 
    pssm_L pssm_M pssm_N pssm_P pssm_Q pssm_R pssm_S pssm_T pssm_V 
    pssm_W pssm_Y pssm_entropy 
    # HMM
    hmm_A hmm_C hmm_D hmm_E hmm_F hmm_G hmm_H hmm_I hmm_K hmm_L 
    hmm_M hmm_N hmm_P hmm_Q hmm_R hmm_S hmm_T hmm_V hmm_W hmm_Y 
    hmm_MM hmm_MI hmm_MD hmm_IM hmm_II hmm_DM hmm_DD hmm_neff 
    # Structural-based features
    rel_sasa ss_helix ss_sheet ss_loop depth hse_up hse_down 
    # DSSP features (float)
    dssp_accessibility dssp_TCO dssp_kappa dssp_alpha dssp_phi dssp_psi 
    # DSSP features (category)
    dssp_sec_struct dssp_helix_3_10 dssp_helix_alpha dssp_helix_pi dssp_helix_pp 
    dssp_bend dssp_chirality dssp_sheet dssp_strand
)

srun python models/DGI/train.py \
     --DATABASE $RRGCL_DATA \
     --config config/DGI.yaml \
     --batch_size 64 \
     --epoch 500 \
     --num_workers 4 \
     --lr 0.001 \
     --patience 50 \
     --min_delta 0.001 \
     --use_scheduler \
     --lr_patience 10 \
     --lr_factor 0.2 \
     --min_lr 1e-6 \
     --wandb_key $WANDB_API_KEY \
     --entity_name $ENTITY_NAME \
     --project_name DGI \
     --wandb_run_name 'Train-1_all' \
     --SAVEPATH $SAVEPATH \
     --node_att "${FEATURES[@]}"