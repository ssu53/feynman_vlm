#!/bin/bash

#SBATCH --job-name=eval_pld
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/nas/ucb/shiyesu/feynman_vlm/slurm_scripts/outputs/%x.%J.out
#SBATCH --error=/nas/ucb/shiyesu/feynman_vlm/slurm_scripts/errors/%x.%J.out
#SBATCH --qos=high
#SBATCH --export=ALL

echo "SLURM_JOBID" $SLURM_JOBID

echo $(pip -V)

srun python3 /nas/ucb/shiyesu/feynman_vlm/run_hf.py