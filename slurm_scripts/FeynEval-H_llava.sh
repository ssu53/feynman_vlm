#!/bin/bash

#SBATCH --job-name=FeynH_llava
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=/nas/ucb/shiyesu/feynman_vlm/slurm_scripts/outputs/%x.%J.out
#SBATCH --error=/nas/ucb/shiyesu/feynman_vlm/slurm_scripts/errors/%x.%J.out
#SBATCH --qos=high
#SBATCH --export=ALL

echo "SLURM_JOBID" $SLURM_JOBID

srun python3 /nas/ucb/shiyesu/feynman_vlm/run_hf.py --model_path llava-hf/llava-1.5-13b-hf --data_index_fn dataset_index/FeynEval-H/data.csv --image_dir dataset/FeynEval-MH-March17/diagrams --out_fn results/FeynEval-H_llava-1.5-13b-hf_ATTEMPT2 --quantise
