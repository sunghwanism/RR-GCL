#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=DS_SAGE
#SBATCH --output=logs/DS_SAGE_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=00:30:00

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

export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True 

export WANDB_API_KEY=$WANDB_API 
export ENTITY_NAME=$WANDB_ENTITY 

FEATURES=(
    copy_idx 
    aa1_KYTJ820101 aa1_KLEP840101 aa1_BHAR880101 aa1_JANJ780101 
    aa1_CHOP780201 aa1_GRAR740102 aa1_GRAR740103 
    pssm_A pssm_C pssm_D pssm_E pssm_F pssm_G pssm_H pssm_I pssm_K 
    pssm_L pssm_M pssm_N pssm_P pssm_Q pssm_R pssm_S pssm_T pssm_V 
    pssm_W pssm_Y pssm_entropy 
    hmm_A hmm_C hmm_D hmm_E hmm_F hmm_G hmm_H hmm_I hmm_K hmm_L 
    hmm_M hmm_N hmm_P hmm_Q hmm_R hmm_S hmm_T hmm_V hmm_W hmm_Y 
    hmm_MM hmm_MI hmm_MD hmm_IM hmm_II hmm_DM hmm_DD hmm_neff 
    rel_sasa ss_helix ss_sheet ss_loop depth hse_up hse_down 
    dssp_accessibility dssp_TCO dssp_kappa dssp_alpha dssp_phi dssp_psi 
    dssp_sec_struct dssp_helix_3_10 dssp_helix_alpha dssp_helix_pi dssp_helix_pp 
    dssp_bend dssp_chirality dssp_sheet dssp_strand
)

srun python models/Classifier/train.py \
     --DATABASE $RRGCL_DATA \
     --config config/DGI.yaml \
     --batch_size 64 \
     --epoch 500 \
     --lr 0.001 \
     --clf_model SAGE \
     --load_model DGI \
     --wandb_key $WANDB_API_KEY \
     --entity_name $ENTITY_NAME \
     --project_name RR-GCL-Classifier \
     --wandb_run_name "Eval_SAGE_Direct" \
     --SAVEPATH $RRGCL_SAVE \
     --node_att "${FEATURES[@]}" \
     --use_scheduler \
     --lr_patience 30 \
     --lr_factor 0.5 \
     --min_lr 1e-6
