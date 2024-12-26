#!/bin/bash

#SBATCH --job-name=test
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=/nas/ucb/shiyesu/feynman_vlm/slurm_scripts/outputs/%x.%J.out
#SBATCH --error=/nas/ucb/shiyesu/feynman_vlm/slurm_scripts/errors/%x.%J.out
#SBATCH --qos=high
#SBATCH --export=ALL

echo "SLURM_JOBID" $SLURM_JOBID

source venv_feynman/bin/activate
python3 --version
which python3
deactivate
