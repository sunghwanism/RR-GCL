#!/bin/bash
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --job-name=ss_processing
#SBATCH --output=logs/ss_processing_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

# load environment variables
source ~/.bashrc

# move to base PATH
cd $WORKING_DIR

# Load module
module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

source $ENV_DIR/rrgcl/bin/activate

# Run the script
srun python data/generation/ss/loadSS.py \
     --input_file $YOUR_EDGE_CSV_FILES \
     --output_file $YOUR_OUTPUT_CSV_FILE \
     --tmp_dir $TEMP_WORKING_DIR \
     --num_workers $YOUR_WORKERS # \
     # --resume_pdb $ADDITIONAL_PDB_TEXT_FILE